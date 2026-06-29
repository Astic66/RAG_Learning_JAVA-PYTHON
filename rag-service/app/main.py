import shutil  # 用于把上传文件流复制到本地磁盘。
from pathlib import Path  # 用于处理文件名和路径。

from fastapi import FastAPI, File, HTTPException, UploadFile  # FastAPI 核心对象和文件上传类型。
from fastapi.middleware.cors import CORSMiddleware  # 允许 Java/前端跨域访问 Python 服务。

from app.config import (
    BASE_DIR, CHAT_MODEL, CHUNK_OVERLAP, CHUNK_SIZE,
    DOCUMENTS_DIR, EMBEDDING_MODEL, TOP_K, VECTOR_STORE_PATH,
)
from app.models import ChatRequest, ChatResponse, DocumentInfo, SourceChunk  # 请求和响应模型。
from app.services.document_loader import load_document_text  # 文档解析：txt/md/pdf/docx -> 纯文本。
from app.services.ollama_client import OllamaClient  # 调用本地 Ollama。
from app.services.text_splitter import split_text  # 文本切块。
from app.services.vector_store import JsonVectorStore  # 本地 JSON 向量库。

app = FastAPI(title="RAG Python Service", version="0.1.0")  # 创建 Python RAG 服务应用。

# 配置 CORS。
# 正常架构下前端访问 Java 8080，Java 再访问 Python 8001。
# 这里允许 8080 访问，是为了调试时也可以直接从页面请求 Python。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保上传文件保存目录存在。
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 创建 Ollama 客户端，后面用于 embedding 和 chat。
ollama = OllamaClient()

# 创建/加载 JSON 向量库。
vector_store = JsonVectorStore(VECTOR_STORE_PATH)


@app.get("/api/health")
def health():
    # 给 Java 和前端检查服务状态用。
    # Java 会把 chat_model/embedding_model 转成 chatModel/embeddingModel 返回给前端。
    return {"status": "ok", "chat_model": CHAT_MODEL, "embedding_model": EMBEDDING_MODEL}


@app.post("/api/documents", response_model=DocumentInfo)
def upload_document(file: UploadFile = File(...)):
    # 1. 获取上传文件后缀，判断是否支持。
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="只支持 txt、md、pdf、docx")

    # 2. 把上传文件保存到 storage/documents。
    saved_path = DOCUMENTS_DIR / Path(file.filename or "uploaded.txt").name
    with saved_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        # 3. 文档解析：把 txt/pdf/docx 统一转成纯文本。
        text = load_document_text(saved_path)

        # 4. 文本切块：长文档切成多个小片段。
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # 5. 如果没有任何文本块，说明文档为空或解析失败。
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容为空，无法建立索引")

        # 6. 对每个文本块调用 embedding 模型，生成向量。
        embeddings = [ollama.embed(EMBEDDING_MODEL, chunk) for chunk in chunks]

        # 7. 把 文本块 + 向量 写入 JSON 向量库。
        document_id = vector_store.add_document(saved_path.name, chunks, embeddings)

        # 8. 返回文档 ID、文件名、切块数量。
        return DocumentInfo(document_id=document_id, file_name=saved_path.name, chunk_count=len(chunks))
    except HTTPException:
        # 主动抛出的 HTTPException 直接交给 FastAPI。
        raise
    except Exception as exc:
        # 其他异常统一包装成 500，方便前端看到错误原因。
        raise HTTPException(status_code=500, detail=f"文档处理失败：{exc}") from exc


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    # 从向量库里按 document_id 聚合文档列表。
    return vector_store.list_documents()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    # 删除某个文档对应的所有向量块。
    deleted = vector_store.delete_document(document_id)
    # 返回删除的文本块数量。
    return {"deleted_chunks": deleted}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 如果向量库为空，说明还没有上传任何文档，无法做 RAG。
    if not vector_store.chunks:
        raise HTTPException(status_code=400, detail="请先上传文档再提问")

    # 1. 把用户问题也转成向量。
    query_embedding = ollama.embed(EMBEDDING_MODEL, request.question)

    # 2. 在向量库里检索和问题最相似的文本块。
    retrieved = vector_store.search(query_embedding, request.top_k or TOP_K)

    # 3. 把检索结果拼成上下文，准备塞给大模型。
    context = "\n\n".join(
        f"资料 {index + 1}：\n来源：{item['document_name']} 第 {item['chunk_index']} 块\n内容：{item['content']}"
        for index, item in enumerate(retrieved)
    )

    # 4. 系统提示词：约束模型必须基于资料回答，避免胡编。
    system_prompt = (
        "你是企业内部知识库问答助手。"
        "你必须优先依据给定资料回答问题。"
        "如果资料中没有答案，请直接说资料中没有找到相关信息，不要编造。"
        "回答要清晰、简洁，并尽量用中文。"
    )

    # 5. 用户提示词：包含问题和检索到的资料。
    user_prompt = f"用户问题：{request.question}\n\n可用资料：\n{context}"

    # 6. 调用聊天模型生成最终答案。
    answer = ollama.chat(CHAT_MODEL, system_prompt, user_prompt)

    # 7. 把检索来源转成响应模型，方便前端展示引用依据。
    sources = [
        SourceChunk(
            document_name=item["document_name"],
            chunk_index=item["chunk_index"],
            score=round(item["score"], 4),
            content=item["content"],
        )
        for item in retrieved
    ]
    # 8. 返回答案和来源。
    return ChatResponse(answer=answer, sources=sources)
