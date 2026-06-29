from pathlib import Path  # 文件路径对象。

from docx import Document as DocxDocument  # 读取 docx 文档。
from pypdf import PdfReader  # 读取 PDF 文档。


# RAG 入库第一步：把不同格式文件统一读成纯文本。
def load_document_text(file_path: Path) -> str:
    # 获取文件后缀，例如 .txt、.pdf、.docx。
    suffix = file_path.suffix.lower()

    # txt 和 md 本来就是文本文件，可以直接读取。
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    # PDF 需要逐页提取文本。
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    # docx 需要读取每个段落的文字。
    if suffix == ".docx":
        doc = DocxDocument(str(file_path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    # 其他格式暂不支持。
    raise ValueError(f"暂不支持的文件类型：{suffix}")
