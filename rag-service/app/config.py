from pathlib import Path  # pathlib 用面向对象方式处理文件路径，比字符串拼路径更清晰。

# 当前 app 目录的上一级，也就是 rag-service 目录。
BASE_DIR = Path(__file__).resolve().parents[1]

# storage 用来保存上传文档和向量索引。
STORAGE_DIR = BASE_DIR / "storage"

# 上传的原始文件保存到 storage/documents。
DOCUMENTS_DIR = STORAGE_DIR / "documents"

# 学习版向量库保存成 JSON 文件。
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.json"

# Ollama 默认本地服务地址。
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# 聊天模型，负责最终回答。
CHAT_MODEL = "qwen2.5:7b"

# 向量模型，负责把文本变成 embedding。
EMBEDDING_MODEL = "nomic-embed-text"

# 每个文本块大约 800 字符。
CHUNK_SIZE = 800

# 相邻文本块重叠 120 字符，避免上下文被切断。
CHUNK_OVERLAP = 120

# 默认召回最相似的 4 个文本块。
TOP_K = 4
