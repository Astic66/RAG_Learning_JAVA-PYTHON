import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VectorChunk:
    id: str
    document_id: str
    document_name: str
    chunk_index: int
    content: str
    embedding: list[float]


class JsonVectorStore:
    """一个学习用 JSON 向量库。

    真实项目会用 Chroma、Milvus、Qdrant、pgvector、Elasticsearch 等向量库。
    这里为了学习透明，把数据直接保存成 JSON，方便你打开看。
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: list[VectorChunk] = []
        self.load()

    def load(self) -> None:
        if not self.store_path.exists():
            self.chunks = []
            return

        raw_items = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.chunks = [VectorChunk(**item) for item in raw_items]

    def save(self) -> None:
        data = [asdict(chunk) for chunk in self.chunks]
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_document(
        self,
        document_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> str:
        document_id = str(uuid.uuid4())
        for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
            self.chunks.append(
                VectorChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=index,
                    content=content,
                    embedding=embedding,
                )
            )
        self.save()
        return document_id

    def list_documents(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for chunk in self.chunks:
            if chunk.document_id not in grouped:
                grouped[chunk.document_id] = {
                    "document_id": chunk.document_id,
                    "file_name": chunk.document_name,
                    "chunk_count": 0,
                }
            grouped[chunk.document_id]["chunk_count"] += 1
        return list(grouped.values())

    def delete_document(self, document_id: str) -> int:
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]
        deleted = before - len(self.chunks)
        self.save()
        return deleted

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """用余弦相似度检索最相关文本块。"""
        if not self.chunks:
            return []

        query_norm = _vector_norm(query_embedding)
        if query_norm == 0:
            return []

        scored = []
        for chunk in self.chunks:
            denominator = query_norm * _vector_norm(chunk.embedding)
            score = 0.0 if denominator == 0 else _dot(query_embedding, chunk.embedding) / denominator
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "score": score,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            for score, chunk in scored[:top_k]
        ]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _vector_norm(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5
