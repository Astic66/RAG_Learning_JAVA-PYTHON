# =============================================================================
# FastAPI 版本配置模块
# =============================================================================
# 本文件集中存放 FastAPI 后端所需的目录路径、模型地址和分块参数。
# 把所有配置放在一个文件里，方便后续修改和维护，避免硬编码散落在各处。
# =============================================================================

from pathlib import Path  # 使用 Path 处理路径，自动适配 Windows 和 Linux


# BASE_DIR 指向 backend 目录。
# __file__ 是当前文件 config.py；.resolve() 取绝对路径；.parents[1] 向上回退两级：
# backend/app/config.py -> backend/app -> backend
BASE_DIR = Path(__file__).resolve().parents[1]

# 数据存储相关目录，全部放在 backend/storage/ 下
STORAGE_DIR = BASE_DIR / "storage"              # 存储根目录
DOCUMENTS_DIR = STORAGE_DIR / "documents"       # 上传的原始文档保存目录
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.json"  # 向量库 JSON 文件路径


# Ollama 本地服务地址，默认监听本机 11434 端口
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# 聊天生成模型：使用本地 qwen2.5:7b，中文问答效果较好
CHAT_MODEL = "qwen2.5:7b"

# 文本嵌入模型：使用 nomic-embed-text，轻量且效果不错
EMBEDDING_MODEL = "nomic-embed-text"


# 文本分块参数
CHUNK_SIZE = 800        # 每个文本块的最大字符数
CHUNK_OVERLAP = 120     # 相邻文本块之间的重叠字符数，保留上下文
TOP_K = 4               # 默认检索时返回的最相关文本块数量
