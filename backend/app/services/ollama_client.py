# =============================================================================
# Ollama 本地模型客户端
# =============================================================================
# 封装对 Ollama 的两个核心接口调用：
#   1. /api/embed  -> 把文本转成向量（Embedding）
#   2. /api/chat   -> 根据上下文生成文本（Chat）
#
# 这里没有使用 LangChain，是为了让学习者直接看到 HTTP 调用细节。
# =============================================================================

import requests  # 第三方 HTTP 库，比标准库 urllib 更简洁

from app.config import OLLAMA_BASE_URL  # 从配置中读取 Ollama 地址


class OllamaClient:
    """Ollama 本地模型客户端。

    这里没有用 LangChain，是为了让你更直观看到：
    1. embedding 接口负责把文本变向量
    2. chat 接口负责根据上下文生成答案
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        """初始化客户端。

        base_url: Ollama 服务地址，例如 http://127.0.0.1:11434
        """
        # rstrip("/") 去掉末尾可能存在的斜杠，方便后续拼接 URL
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> list[float]:
        """调用 Ollama embedding 模型，把文本转成向量。

        model: 模型名称，例如 "nomic-embed-text"
        text:  要向量化的文本
        """
        # 构造请求体：指定模型和输入文本
        payload = {"model": model, "input": text}

        response = requests.post(
            f"{self.base_url}/api/embed",  # Ollama 新版 embedding 接口
            json=payload,
            timeout=120,                    # 向量化可能较慢，设置 120 秒超时
        )

        # 兼容处理：如果新版 /api/embed 返回 404，可能是旧版 Ollama
        if response.status_code == 404:
            # 旧版 Ollama 使用 /api/embeddings，参数名为 prompt
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()     # 如果仍然失败，抛出异常
            return response.json()["embedding"]

        response.raise_for_status()         # 检查 HTTP 错误
        data = response.json()

        # 新版接口返回 embeddings 数组，因为只传一个文本，取第 0 个
        return data["embeddings"][0]

    def chat(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """调用 Ollama 聊天模型生成答案。

        model:          聊天模型名称，例如 "qwen2.5:7b"
        system_prompt:  系统提示词，定义助手行为
        user_prompt:    用户提示词，包含问题和参考资料
        """
        response = requests.post(
            f"{self.base_url}/api/chat",    # Ollama 聊天接口
            json={
                "model": model,             # 指定模型
                "stream": False,            # 非流式，一次性返回完整答案
                "messages": [               # OpenAI 风格消息列表
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=180,                    # 生成答案可能更慢，设置 180 秒超时
        )

        response.raise_for_status()         # 检查 HTTP 错误

        # 提取模型返回的消息内容
        return response.json()["message"]["content"]
