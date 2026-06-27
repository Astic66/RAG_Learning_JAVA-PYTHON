from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


def load_document_text(file_path: Path) -> str:
    """读取上传文件内容，并统一转成纯文本。

    RAG 的第一步是“文档加载”。
    不管原始文件是 txt、pdf、docx，后续切块和向量化都需要拿到纯文本。
    """
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if suffix == ".docx":
        doc = DocxDocument(str(file_path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    raise ValueError(f"暂不支持的文件类型：{suffix}")
