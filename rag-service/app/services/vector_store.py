import json  # 读写 JSON 向量库文件。
import uuid  # 生成 document_id 和 chunk_id。
from dataclasses import dataclass, field, asdict  # 用 dataclass 定义向量块结构。
from pathlib import Path  # 文件路径对象。


# 一个文本块在向量库里的结构。
@dataclass
class VectorChunk:
    chunk_id: str  # 文本块唯一 ID。
    document_id: str  # 所属文档 ID。
    document_name: str  # 来源文档名。
    chunk_index: int  # 在文档中的第几个块。
    content: str  # 文本块原文。
    embedding: list[float] = field(default_factory=list)  # 文本块向量。


# 两个向量点乘。
def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# 计算向量长度，也叫 L2 norm。
def _vector_norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


# 学习版 JSON 向量库。
# 真实生产项目一般会换成 Milvus、Qdrant、Chroma、pgvector、Elasticsearch 等。
class JsonVectorStore:
    # 初始化时传入 vector_store.json 的路径。
    def __init__(self, path: Path):
        # 保存向量库文件路径。
        self.path = Path(path)
        # 内存里的所有文本块。
        self.chunks: list[VectorChunk] = []
        # 启动时从 JSON 文件加载已有索引。
        self.load()

    # 从磁盘读取向量库。
    def load(self) -> None:
        # 如果文件不存在，说明还没有任何索引。
        if not self.path.exists():
            self.chunks = []
            return
        try:
            # 读取 JSON 文件。
            data = json.loads(self.path.read_text(encoding="utf-8"))
            # 把字典转回 VectorChunk 对象。
            self.chunks = [VectorChunk(**item) for item in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            # 如果 JSON 损坏或字段不对，就清空，避免服务启动失败。
            self.chunks = []

    # 把内存里的向量块写回磁盘。
    def save(self) -> None:
        # 确保存储目录存在。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # dataclass 转普通 dict，方便 json.dumps。
        data = [asdict(chunk) for chunk in self.chunks]
        # ensure_ascii=False 可以保存中文，不会变成 unicode 转义。
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 添加一个文档的所有切块和向量。
    def add_document(
        self,
        document_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> str:
        # 每次上传文档都生成一个新的 document_id。
        document_id = str(uuid.uuid4())
        # zip 把文本块和对应 embedding 一一配对。
        for index, (text, embedding) in enumerate(zip(chunks, embeddings)):
            # 每个 chunk 都保存为一条 VectorChunk。
            self.chunks.append(
                VectorChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=index,
                    content=text,
                    embedding=embedding,
                )
            )
        # 添加完成后保存到 JSON 文件。
        self.save()
        # 返回文档 ID 给前端展示/删除使用。
        return document_id

    # 聚合文档列表。
    def list_documents(self) -> list[dict]:
        # doc_map 用 document_id 分组统计 chunk_count。
        doc_map: dict[str, dict] = {}
        for chunk in self.chunks:
            if chunk.document_id not in doc_map:
                doc_map[chunk.document_id] = {
                    "document_id": chunk.document_id,
                    "file_name": chunk.document_name,
                    "chunk_count": 0,
                }
            doc_map[chunk.document_id]["chunk_count"] += 1
        # 返回每个文档的信息。
        return list(doc_map.values())

    # 删除某个文档的所有文本块。
    def delete_document(self, document_id: str) -> int:
        # 删除前的总块数。
        original_count = len(self.chunks)
        # 过滤掉指定 document_id 的块。
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        # 计算删除了多少块。
        deleted = original_count - len(self.chunks)
        # 如果确实删了内容，就保存。
        if deleted:
            self.save()
        # 返回删除块数量。
        return deleted

    # 语义检索：根据问题向量，找最相似的文本块。
    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        # 先计算问题向量长度。
        query_norm = _vector_norm(query_embedding)
        # 零向量没有意义，直接返回空。
        if query_norm == 0:
            return []

        # 保存每个文本块的相似度分数。
        scored: list[tuple[float, VectorChunk]] = []
        for chunk in self.chunks:
            # 计算文本块向量长度。
            chunk_norm = _vector_norm(chunk.embedding)
            # 跳过异常零向量。
            if chunk_norm == 0:
                continue
            # 余弦相似度：两个向量越接近，分数越高。
            score = _dot(query_embedding, chunk.embedding) / (query_norm * chunk_norm)
            scored.append((score, chunk))

        # 按相似度从高到低排序。
        scored.sort(key=lambda item: item[0], reverse=True)
        # 组装返回给 main.py 的结果。
        results = []
        for score, chunk in scored[:top_k]:
            results.append(
                {
                    "document_name": chunk.document_name,
                    "chunk_index": chunk.chunk_index,
                    "score": float(score),
                    "content": chunk.content,
                }
            )
        # 返回 top_k 个最相关片段。
        return results
