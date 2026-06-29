# RAG 入库第二步：文本切块。
def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    # 去掉空行，并把所有非空行重新拼成一个长文本。
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    # 如果文档没有内容，直接返回空列表。
    if not cleaned:
        return []

    # 保存切出来的所有文本块。
    chunks: list[str] = []
    # start 表示当前切块起始位置。
    start = 0
    # 文本总长度。
    text_length = len(cleaned)

    # 只要还没切到文本末尾，就继续切。
    while start < text_length:
        # 当前块结束位置，不能超过文本总长度。
        end = min(start + chunk_size, text_length)
        # 截取当前文本块。
        chunk = cleaned[start:end].strip()
        # 如果块不为空，就加入结果。
        if chunk:
            chunks.append(chunk)
        # 已经切到末尾就结束循环。
        if end == text_length:
            break
        # 下一块从 end - overlap 开始，保留一部分重叠上下文。
        start = max(end - chunk_overlap, start + 1)

    # 返回所有文本块。
    return chunks
