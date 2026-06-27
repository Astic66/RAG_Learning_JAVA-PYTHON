import json
import mimetypes
import shutil
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
STORAGE_DIR = BASE_DIR / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.json"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_store() -> list[dict]:
    if not VECTOR_STORE_PATH.exists():
        return []
    return json.loads(VECTOR_STORE_PATH.read_text(encoding="utf-8"))


def save_store(items: list[dict]) -> None:
    VECTOR_STORE_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def split_text(text: str) -> list[str]:
    """文本切块：RAG 入库前必须把长文档切成较小片段。"""
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + CHUNK_SIZE, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def ollama_post(path: str, payload: dict, timeout: int = 180) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def embed_text(text: str) -> list[float]:
    """调用本地 embedding 模型，把文本变成向量。"""
    data = ollama_post("/api/embed", {"model": EMBEDDING_MODEL, "input": text})
    return data["embeddings"][0]


def chat_with_context(question: str, context: str) -> str:
    """调用本地聊天模型，让它基于检索上下文回答。"""
    system_prompt = (
        "你是企业内部知识库问答助手。"
        "你必须优先依据给定资料回答问题。"
        "如果资料中没有答案，请直接说资料中没有找到相关信息，不要编造。"
        "回答要清晰、简洁，并使用中文。"
    )
    user_prompt = f"用户问题：{question}\n\n可用资料：\n{context}"
    data = ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    return data["message"]["content"]


def vector_norm(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denominator = vector_norm(left) * vector_norm(right)
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def search_chunks(question: str, top_k: int = 4) -> list[dict]:
    """问题向量化后，在本地 JSON 向量库中做相似度检索。"""
    store = load_store()
    query_embedding = embed_text(question)
    scored = []
    for item in store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, item in scored[:top_k]:
        copied = dict(item)
        copied["score"] = round(score, 4)
        results.append(copied)
    return results


def parse_multipart_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    """解析浏览器 FormData 上传的单个文件。

    这是学习项目的简化版 multipart 解析，只处理一个 file 字段。
    """
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("缺少 multipart boundary")

    boundary = ("--" + content_type.split(marker, 1)[1]).encode("utf-8")
    parts = body.split(boundary)

    for part in parts:
        if b'Content-Disposition:' not in part or b'filename="' not in part:
            continue

        header, file_content = part.split(b"\r\n\r\n", 1)
        header_text = header.decode("utf-8", errors="ignore")
        filename = header_text.split('filename="', 1)[1].split('"', 1)[0]
        file_content = file_content.rstrip(b"\r\n-")
        return filename, file_content

    raise ValueError("没有找到上传文件")


class RagHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict | list, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"detail": message}, status)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "chat_model": CHAT_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                }
            )
            return

        if path == "/api/documents":
            grouped = {}
            for item in load_store():
                doc_id = item["document_id"]
                grouped.setdefault(
                    doc_id,
                    {
                        "document_id": doc_id,
                        "file_name": item["document_name"],
                        "chunk_count": 0,
                    },
                )
                grouped[doc_id]["chunk_count"] += 1
            self._send_json(list(grouped.values()))
            return

        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            if path == "/api/documents":
                self._upload_document(body)
                return

            if path == "/api/chat":
                payload = json.loads(body.decode("utf-8"))
                self._chat(payload)
                return

            self._send_error("接口不存在", 404)
        except Exception as exc:
            self._send_error(str(exc), 500)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        prefix = "/api/documents/"
        if not path.startswith(prefix):
            self._send_error("接口不存在", 404)
            return

        document_id = unquote(path[len(prefix):])
        store = load_store()
        new_store = [item for item in store if item["document_id"] != document_id]
        save_store(new_store)
        self._send_json({"deleted_chunks": len(store) - len(new_store)})

    def _upload_document(self, body: bytes) -> None:
        filename, file_content = parse_multipart_file(
            self.headers.get("Content-Type", ""),
            body,
        )
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise ValueError("当前零依赖版本先支持 txt、md。PDF/DOCX 后面可以再接解析库。")

        safe_name = Path(filename).name
        saved_path = DOCUMENTS_DIR / safe_name
        saved_path.write_bytes(file_content)

        text = file_content.decode("utf-8", errors="ignore")
        chunks = split_text(text)
        if not chunks:
            raise ValueError("文档内容为空")

        document_id = str(uuid.uuid4())
        store = load_store()
        for index, chunk in enumerate(chunks):
            store.append(
                {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "document_name": safe_name,
                    "chunk_index": index,
                    "content": chunk,
                    "embedding": embed_text(chunk),
                }
            )
        save_store(store)
        self._send_json(
            {
                "document_id": document_id,
                "file_name": safe_name,
                "chunk_count": len(chunks),
            }
        )

    def _chat(self, payload: dict) -> None:
        question = (payload.get("question") or "").strip()
        top_k = int(payload.get("top_k") or 4)
        if not question:
            raise ValueError("问题不能为空")

        if not load_store():
            self._send_error("请先上传文档再提问")
            return

        results = search_chunks(question, top_k)
        context = "\n\n".join(
            f"资料 {index + 1}：\n来源：{item['document_name']} 第 {item['chunk_index']} 块\n内容：{item['content']}"
            for index, item in enumerate(results)
        )
        answer = chat_with_context(question, context)

        self._send_json(
            {
                "answer": answer,
                "sources": [
                    {
                        "document_name": item["document_name"],
                        "chunk_index": item["chunk_index"],
                        "score": item["score"],
                        "content": item["content"],
                    }
                    for item in results
                ],
            }
        )

    def _serve_static(self, path: str) -> None:
        if path == "/":
            file_path = FRONTEND_DIR / "index.html"
        elif path.startswith("/assets/"):
            file_path = FRONTEND_DIR / path.replace("/assets/", "", 1)
        else:
            file_path = FRONTEND_DIR / path.lstrip("/")

        if not file_path.exists() or not file_path.is_file():
            self._send_error("页面不存在", 404)
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.end_headers()
        with file_path.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), RagHandler)
    print("RAG 企业内部知识问答系统已启动：http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
