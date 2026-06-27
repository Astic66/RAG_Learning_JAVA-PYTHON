# =============================================================================
# FastAPI 版本后端入口
# =============================================================================
# 这是 RAG 后端的 FastAPI 实现版本，功能与 server.py 完全一致，
# 但使用 FastAPI 框架提供更工程化的路由、参数校验、自动文档和文件上传处理。
# 同时支持 txt、md、pdf、docx 四种文档格式。
# =============================================================================

import shutil                 # 用于把上传文件流拷贝到本地磁盘
from pathlib import Path      # 路径操作

# FastAPI 核心组件
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware   # 跨域中间件
from fastapi.responses import FileResponse           # 返回文件响应
from fastapi.staticfiles import StaticFiles          # 挂载静态文件目录

# 从配置模块导入所有常量
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

# 从模型模块导入 Pydantic 模型
from app.models import ChatRequest, ChatResponse, DocumentInfo, SourceChunk

# 从服务层导入业务逻辑
from app.services.document_loader import load_document_text   # 文档解析
from app.services.ollama_client import OllamaClient           # Ollama 客户端
from app.services.text_splitter import split_text             # 文本分块
from app.services.vector_store import JsonVectorStore         # JSON 向量库


# 创建 FastAPI 应用实例，设置标题和版本
app = FastAPI(title="RAG 企业内部知识问答系统", version="0.1.0")

# 添加 CORS 中间件，允许前端（运行在不同端口或本地文件时）访问后端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源，开发环境常用
    allow_credentials=True,    # 允许携带 cookie 等凭证
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
)


# 前端目录位于项目根目录下的 frontend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# 确保文档保存目录存在
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 初始化 Ollama 客户端和向量库实例
ollama = OllamaClient()
vector_store = JsonVectorStore(VECTOR_STORE_PATH)

# 如果前端目录存在，把 /assets 路径挂载为静态文件服务
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


# ---------------------------------------------------------------------------
# 路由定义
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """根路径返回前端首页。"""
    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)  # 返回 index.html 文件

    # 如果前端文件不存在，返回简单的提示信息
    return {"message": "RAG 企业内部知识问答系统后端已启动"}


@app.get("/api/health")
def health():
    """健康检查接口，返回当前使用的模型名称。"""
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
    # 获取文件后缀并转小写
    suffix = Path(file.filename or "").suffix.lower()

    # 校验文件类型
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="只支持 txt、md、pdf、docx")

    # 拼接保存路径，确保文件名安全
    saved_path = DOCUMENTS_DIR / Path(file.filename or "uploaded.txt").name

    # 把上传文件流写入本地磁盘
    with saved_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        # 调用文档加载器，把不同格式转成纯文本
        text = load_document_text(saved_path)

        # 对文本进行分块
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容为空，无法建立索引")

        # 对每个文本块调用 Embedding 模型生成向量
        embeddings = [ollama.embed(EMBEDDING_MODEL, chunk) for chunk in chunks]

        # 把文档名、文本块、向量一起存入向量库
        document_id = vector_store.add_document(saved_path.name, chunks, embeddings)

        return DocumentInfo(
            document_id=document_id,
            file_name=saved_path.name,
            chunk_count=len(chunks),
        )

    except HTTPException:
        # 如果是我们主动抛出的 HTTPException，直接继续向上抛
        raise
    except Exception as exc:
        # 其他未知异常统一包装成 500 错误
        raise HTTPException(status_code=500, detail=f"文档处理失败：{exc}") from exc


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    """列出所有已上传文档及其块数。"""
    return vector_store.list_documents()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    """删除指定文档的所有向量索引。"""
    deleted = vector_store.delete_document(document_id)
    return {"deleted_chunks": deleted}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """提问接口。

    这就是 RAG 的“问答”流程：
    用户问题 -> 问题向量化 -> 向量库检索 -> 拼上下文 -> 大模型回答
    """
    # 如果向量库为空，提示先上传文档
    if not vector_store.chunks:
        raise HTTPException(status_code=400, detail="请先上传文档再提问")

    # 把用户问题转成向量
    query_embedding = ollama.embed(EMBEDDING_MODEL, request.question)

    # 在向量库中检索最相关的 top_k 个文本块
    retrieved = vector_store.search(query_embedding, request.top_k or TOP_K)

    # 把检索到的片段拼接成上下文字符串
    context = "\n\n".join(
        f"资料 {index + 1}：\n来源：{item['document_name']} 第 {item['chunk_index']} 块\n内容：{item['content']}"
        for index, item in enumerate(retrieved)
    )

    # 系统提示词，约束模型基于资料回答、不编造
    system_prompt = (
        "你是企业内部知识库问答助手。"
        "你必须优先依据给定资料回答问题。"
        "如果资料中没有答案，请直接说资料中没有找到相关信息，不要编造。"
        "回答要清晰、简洁，并尽量用中文。"
    )

    # 拼接最终的用户提示词
    user_prompt = f"用户问题：{request.question}\n\n可用资料：\n{context}"

    # 调用聊天模型生成答案
    answer = ollama.chat(CHAT_MODEL, system_prompt, user_prompt)

    # 把检索结果转换成 SourceChunk 模型列表
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
