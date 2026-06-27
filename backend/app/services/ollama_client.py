import requests

from app.config import OLLAMA_BASE_URL


class OllamaClient:
    """Ollama 本地模型客户端。

    这里没有用 LangChain，是为了让你更直观看到：
    1. embedding 接口负责把文本变向量
    2. chat 接口负责根据上下文生成答案
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> list[float]:
        """调用 Ollama embedding 模型，把文本转成向量。"""
        payload = {"model": model, "input": text}
        response = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=120,
        )

        if response.status_code == 404:
            # 兼容旧版 Ollama 的 embeddings 接口。
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["embedding"]

        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]

    def chat(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """调用 Ollama 聊天模型生成答案。"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
