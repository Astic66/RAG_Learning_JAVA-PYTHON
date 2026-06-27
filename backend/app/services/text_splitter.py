def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """把长文档切成多个文本块。

    为什么要切块：
    1. 大模型上下文有限，不能把整本资料一次性塞进去。
    2. 向量检索需要较小的语义单元，太长会影响匹配精度。
    3. overlap 可以保留上下文，避免一句话刚好被切断。
    """
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks
