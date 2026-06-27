import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    BASE_DIR,
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    VECTOR_STORE_PATH,
)
from app.models import ChatRequest, ChatResponse, DocumentInfo, SourceChunk
from app.services.document_loader import load_document_text
from app.services.ollama_client import OllamaClient
from app.services.text_splitter import split_text
from app.services.vector_store import JsonVectorStore


app = FastAPI(title="RAG 企业内部知识问答系统", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR.parent / "frontend"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

ollama = OllamaClient()
vector_store = JsonVectorStore(VECTOR_STORE_PATH)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "RAG 企业内部知识问答系统后端已启动"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/api/documents", response_model=DocumentInfo)
def upload_document(file: UploadFile = File(...)):
    """上传文档，并建立向量索引。

    这就是 RAG 的“入库”流程：
    文件 -> 文本解析 -> 文本切块 -> embedding -> 保存向量
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="只支持 txt、md、pdf、docx")

    saved_path = DOCUMENTS_DIR / Path(file.filename or "uploaded.txt").name
    with saved_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        text = load_document_text(saved_path)
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容为空，无法建立索引")

        embeddings = [ollama.embed(EMBEDDING_MODEL, chunk) for chunk in chunks]
        document_id = vector_store.add_document(saved_path.name, chunks, embeddings)
        return DocumentInfo(
            document_id=document_id,
            file_name=saved_path.name,
            chunk_count=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文档处理失败：{exc}") from exc


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    return vector_store.list_documents()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    deleted = vector_store.delete_document(document_id)
    return {"deleted_chunks": deleted}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """提问接口。

    这就是 RAG 的“问答”流程：
    用户问题 -> 问题向量化 -> 向量库检索 -> 拼上下文 -> 大模型回答
    """
    if not vector_store.chunks:
        raise HTTPException(status_code=400, detail="请先上传文档再提问")

    query_embedding = ollama.embed(EMBEDDING_MODEL, request.question)
    retrieved = vector_store.search(query_embedding, request.top_k or TOP_K)

    context = "\n\n".join(
        f"资料 {index + 1}：\n来源：{item['document_name']} 第 {item['chunk_index']} 块\n内容：{item['content']}"
        for index, item in enumerate(retrieved)
    )

    system_prompt = (
        "你是企业内部知识库问答助手。"
        "你必须优先依据给定资料回答问题。"
        "如果资料中没有答案，请直接说资料中没有找到相关信息，不要编造。"
        "回答要清晰、简洁，并尽量用中文。"
    )
    user_prompt = f"用户问题：{request.question}\n\n可用资料：\n{context}"
    answer = ollama.chat(CHAT_MODEL, system_prompt, user_prompt)

    sources = [
        SourceChunk(
            document_name=item["document_name"],
            chunk_index=item["chunk_index"],
            score=round(item["score"], 4),
            content=item["content"],
        )
        for item in retrieved
    ]
    return ChatResponse(answer=answer, sources=sources)
