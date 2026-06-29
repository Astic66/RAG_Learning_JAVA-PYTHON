from pydantic import BaseModel, Field  # FastAPI 用 Pydantic 做请求/响应数据校验。


# 前端/Java 调 Python /api/chat 时发送的请求体。
class ChatRequest(BaseModel):
    # question 是用户问题，不能为空。
    question: str = Field(..., min_length=1, description="用户问题")
    # top_k 表示检索多少个相关文本块，限制在 1 到 10 之间。
    top_k: int = Field(default=4, ge=1, le=10, description="召回片段数量")


# 一个 RAG 来源片段，用于告诉前端“答案依据来自哪里”。
class SourceChunk(BaseModel):
    # 来源文档名。
    document_name: str
    # 文档切块编号。
    chunk_index: int
    # 相似度分数，越高表示和问题越相关。
    score: float
    # 文本块内容。
    content: str


# Python /api/chat 返回的数据。
class ChatResponse(BaseModel):
    # 大模型生成的最终答案。
    answer: str
    # 检索出来的来源片段列表。
    sources: list[SourceChunk]


# 上传文档或查询文档列表时返回的数据。
class DocumentInfo(BaseModel):
    # 文档唯一 ID。
    document_id: str
    # 文件名。
    file_name: str
    # 被切成多少个文本块。
    chunk_count: int
