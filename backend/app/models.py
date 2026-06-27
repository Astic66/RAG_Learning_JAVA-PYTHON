# =============================================================================
# FastAPI 版本数据模型定义
# =============================================================================
# 使用 Pydantic 定义请求体和响应体的结构。
# 这样做有两个好处：
#   1. FastAPI 会自动校验请求参数类型，减少手写校验代码。
#   2. 自动生成交互式 API 文档（/docs）。
# =============================================================================

from pydantic import BaseModel, Field  # BaseModel 是模型基类，Field 用于字段约束


class ChatRequest(BaseModel):
    """用户提问时的请求体模型。"""

    # question 是必填字段，最小长度 1，防止空问题
    question: str = Field(..., min_length=1, description="用户问题")

    # top_k 是可选字段，默认 4，必须在 1~10 之间
    top_k: int = Field(default=4, ge=1, le=10, description="召回片段数量")


class SourceChunk(BaseModel):
    """单个来源片段的响应模型。"""

    document_name: str   # 来源文档名称
    chunk_index: int     # 该片段在文档中的序号
    score: float         # 与问题的相似度分数
    content: str         # 片段文本内容


class ChatResponse(BaseModel):
    """提问接口的响应体模型。"""

    answer: str                  # 模型生成的答案
    sources: list[SourceChunk]   # 答案所引用的来源片段列表


class DocumentInfo(BaseModel):
    """文档信息模型，用于上传后和列表接口返回。"""

    document_id: str    # 文档唯一 ID
    file_name: str      # 原始文件名
    chunk_count: int    # 该文档被切成了多少个文本块
