# =============================================================================
# JSON 向量库（Vector Store）
# =============================================================================
# 用纯 JSON 文件保存文本块的向量和元数据，并提供增删改查接口。
#
# 真实项目会用 Chroma、Milvus、Qdrant、pgvector、Elasticsearch 等专业向量库。
# 这里用 JSON 是为了学习透明，方便直接打开文件看数据。
# =============================================================================

import json                     # 序列化/反序列化 JSON
import uuid                     # 生成唯一 ID
from dataclasses import asdict, dataclass  # dataclass 简化数据类定义
from pathlib import Path        # 路径操作


@dataclass
class VectorChunk:
    """向量块数据类，代表一个文本块及其向量。"""

    id: str             # 块唯一 ID
    document_id: str    # 所属文档 ID
    document_name: str  # 所属文档名称
    chunk_index: int    # 块在文档中的序号
    content: str        # 块文本内容
    embedding: list[float]  # 块的向量表示


class JsonVectorStore:
    """一个学习用 JSON 向量库。

    真实项目会用 Chroma、Milvus、Qdrant、pgvector、Elasticsearch 等向量库。
    这里为了学习透明，把数据直接保存成 JSON，方便你打开看。
    """

    def __init__(self, store_path: Path):
        """初始化向量库。

        store_path: JSON 向量库文件路径
        """
        self.store_path = store_path

        # 如果存储目录不存在，自动创建
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # 内存中保存所有向量块的列表
        self.chunks: list[VectorChunk] = []

        # 初始化时加载已有数据
        self.load()

    def load(self) -> None:
        """从 JSON 文件加载所有向量块到内存。"""
        if not self.store_path.exists():   # 如果文件不存在，初始化为空列表
            self.chunks = []
            return

        # 读取 JSON 文件并解析
        raw_items = json.loads(self.store_path.read_text(encoding="utf-8"))

        # 把每个字典反序列化成 VectorChunk 对象
        self.chunks = [VectorChunk(**item) for item in raw_items]

    def save(self) -> None:
        """把内存中的向量块写回 JSON 文件。"""
        # 把 VectorChunk 对象列表转成字典列表
        data = [asdict(chunk) for chunk in self.chunks]

        # ensure_ascii=False 保证中文可读，indent=2 格式化便于查看
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
        """添加一个文档的所有文本块到向量库。

        document_name: 文档原始文件名
        chunks:        文本块内容列表
        embeddings:    与 chunks 一一对应的向量列表

        返回：生成的文档唯一 ID
        """
        document_id = str(uuid.uuid4())  # 为文档生成唯一 ID

        # 使用 zip 同时遍历 chunks 和 embeddings，保证一一对应
        for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
            self.chunks.append(
                VectorChunk(
                    id=str(uuid.uuid4()),           # 每个块一个唯一 ID
                    document_id=document_id,        # 属于该文档
                    document_name=document_name,    # 文档名
                    chunk_index=index,              # 块序号
                    content=content,                # 块文本
                    embedding=embedding,            # 块向量
                )
            )

        self.save()                      # 保存到文件
        return document_id               # 返回文档 ID

    def list_documents(self) -> list[dict]:
        """列出所有文档及其包含的块数量。"""
        grouped: dict[str, dict] = {}    # 用字典按 document_id 分组

        for chunk in self.chunks:        # 遍历所有块
            if chunk.document_id not in grouped:
                # 第一次遇到该文档，初始化信息
                grouped[chunk.document_id] = {
                    "document_id": chunk.document_id,
                    "file_name": chunk.document_name,
                    "chunk_count": 0,
                }

            grouped[chunk.document_id]["chunk_count"] += 1  # 累加块数

        return list(grouped.values())    # 把字典转成列表返回

    def delete_document(self, document_id: str) -> int:
        """删除指定文档的所有向量块。

        document_id: 要删除的文档 ID
        返回：实际删除的块数量
        """
        before = len(self.chunks)        # 删除前总块数

        # 过滤掉属于该 document_id 的块
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]

        deleted = before - len(self.chunks)  # 计算删除了多少块
        self.save()                          # 保存更新后的结果
        return deleted

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """用余弦相似度检索最相关文本块。

        query_embedding: 用户问题的向量
        top_k:           返回最相关的块数量
        """
        if not self.chunks:              # 如果库为空，直接返回空列表
            return []

        query_norm = _vector_norm(query_embedding)  # 计算问题向量的模长
        if query_norm == 0:              # 防止除零
            return []

        scored = []                      # 保存 (相似度, 块) 二元组

        for chunk in self.chunks:        # 遍历所有块
            denominator = query_norm * _vector_norm(chunk.embedding)  # 分母

            if denominator == 0:
                score = 0.0              # 模长为 0 时相似度设为 0
            else:
                # 余弦相似度 = 点积 / (两个向量模长乘积)
                score = _dot(query_embedding, chunk.embedding) / denominator

            scored.append((score, chunk))

        # 按相似度从高到低排序
        scored.sort(key=lambda item: item[0], reverse=True)

        # 取前 top_k 个，并转换成字典列表返回
        return [
            {
                "score": score,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            for score, chunk in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# 向量计算工具函数
# ---------------------------------------------------------------------------
def _dot(left: list[float], right: list[float]) -> float:
    """计算两个向量的点积。"""
    return sum(a * b for a, b in zip(left, right))


def _vector_norm(vector: list[float]) -> float:
    """计算向量的 L2 范数（模长）。"""
    return sum(value * value for value in vector) ** 0.5
