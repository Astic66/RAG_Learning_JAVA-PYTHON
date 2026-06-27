from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.json"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
