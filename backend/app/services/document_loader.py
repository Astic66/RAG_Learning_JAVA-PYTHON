# =============================================================================
# 文档加载器（Document Loader）
# =============================================================================
# 负责把用户上传的不同格式文件（txt/md/pdf/docx）统一解析成纯文本。
# RAG 的后续步骤（分块、向量化）只处理字符串，因此这里必须把各种文件“标准化”成文本。
# =============================================================================

from pathlib import Path  # 路径操作

# 第三方库，用于解析 Word 和 PDF 文档
from docx import Document as DocxDocument   # python-docx：读取 .docx
from pypdf import PdfReader                  # pypdf：读取 .pdf


def load_document_text(file_path: Path) -> str:
    """读取上传文件内容，并统一转成纯文本。

    RAG 的第一步是“文档加载”。
    不管原始文件是 txt、pdf、docx，后续切块和向量化都需要拿到纯文本。
    """
    # 获取文件后缀并统一转小写，便于判断文件类型
    suffix = file_path.suffix.lower()

    # 处理纯文本文件：txt 和 markdown
    if suffix in {".txt", ".md"}:
        # 按 UTF-8 读取，遇到编码错误时忽略，避免某些文件导致程序崩溃
        return file_path.read_text(encoding="utf-8", errors="ignore")

    # 处理 PDF 文件
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))  # pypdf 需要字符串路径

        # 遍历每一页，提取文本；某些扫描件页面可能提取不到文本，用空字符串兜底
        pages = [page.extract_text() or "" for page in reader.pages]

        # 把所有页文本用换行符拼接成完整字符串
        return "\n".join(pages)

    # 处理 Word 文档
    if suffix == ".docx":
        doc = DocxDocument(str(file_path))  # python-docx 需要字符串路径

        # 遍历文档中的每个段落，提取文本后用换行符拼接
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    # 如果是不支持的格式，抛出明确错误
    raise ValueError(f"暂不支持的文件类型：{suffix}")
