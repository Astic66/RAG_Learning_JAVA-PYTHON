# =============================================================================
# 文本分块器（Text Splitter）
# =============================================================================
# 负责把长文档切分成较小的文本块，以便后续向量化、检索和输入大模型。
#
# 为什么要分块：
#   1. 大模型上下文有限，不能一次性塞入整篇长文档。
#   2. 向量检索需要较小的语义单元，块太大或太小都会影响匹配精度。
#   3. 相邻块之间保留重叠（overlap），可以避免句子在边界被切断。
# =============================================================================


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """把长文档切成多个文本块。

    参数：
        text:          原始长文本
        chunk_size:    每个块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    实现逻辑：
        1. 先清洗文本，去掉空行和每行首尾空白。
        2. 使用滑动窗口遍历文本，窗口大小为 chunk_size。
        3. 每次前进 chunk_size - chunk_overlap 个字符，保证相邻块有重叠。
    """
    # 清洗文本：
    # 1. text.splitlines() 按行拆分
    # 2. 去掉空行（if line.strip()）
    # 3. 去掉每行首尾空白（line.strip()）
    # 4. 用换行符重新拼接成干净文本
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    # 如果清洗后文本为空，直接返回空列表
    if not cleaned:
        return []

    chunks: list[str] = []      # 用于保存切出来的文本块
    start = 0                   # 当前窗口起始位置
    text_length = len(cleaned)  # 文本总长度

    while start < text_length:                         # 只要窗口起点还在文本范围内就继续
        end = min(start + chunk_size, text_length)     # 窗口终点，不超过文本末尾
        chunk = cleaned[start:end].strip()             # 取出当前窗口内容并去首尾空

        if chunk:                                      # 非空才加入结果
            chunks.append(chunk)

        if end == text_length:                         # 如果已经到达文本末尾，结束循环
            break

        # 计算下一个窗口起点：
        # 终点往前回退 chunk_overlap 个字符，保证重叠。
        # max(..., start + 1) 是为了防止 overlap 过大导致死循环，确保至少前进 1 个字符。
        start = max(end - chunk_overlap, start + 1)

    return chunks
