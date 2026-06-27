from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=10, description="召回片段数量")


class SourceChunk(BaseModel):
    document_name: str
    chunk_index: int
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    chunk_count: int
