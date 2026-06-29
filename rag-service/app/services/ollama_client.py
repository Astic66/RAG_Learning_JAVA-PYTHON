import requests  # 用 HTTP 请求调用 Ollama 本地接口。

from app.config import OLLAMA_BASE_URL  # Ollama 服务地址。


# 封装 Ollama API。
# Python 服务和 Ollama 的通信方式也是 HTTP：
# embedding：POST http://127.0.0.1:11434/api/embed
# chat：POST http://127.0.0.1:11434/api/chat
class OllamaClient:
    # 初始化客户端，可以传自定义 base_url，不传则使用配置里的本地 Ollama 地址。
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        # 去掉结尾的 /，避免拼 URL 时出现 //。
        self.base_url = base_url.rstrip("/")

    # 把文本转成向量。
    # RAG 入库时：文档块 -> embedding。
    # RAG 查询时：用户问题 -> embedding。
    def embed(self, model: str, text: str) -> list[float]:
        # 新版 Ollama /api/embed 的请求格式。
        payload = {"model": model, "input": text}
        # 调用本地 Ollama embedding 接口。
        response = requests.post(
            f"{self.base_url}/api/embed", json=payload, timeout=120
        )
        # 如果本机 Ollama 版本比较旧，可能没有 /api/embed。
        if response.status_code == 404:
            # 兼容旧接口 /api/embeddings。
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=120,
            )
            # 如果请求失败，抛出异常，让 FastAPI 返回错误。
            response.raise_for_status()
            # 旧接口返回字段叫 embedding。
            return response.json()["embedding"]
        # 新接口请求失败也抛异常。
        response.raise_for_status()
        # 新接口返回字段是 embeddings，是二维数组。
        data = response.json()
        # 这里只传了一段文本，所以取第一个向量。
        return data["embeddings"][0]

    # 调用聊天模型生成答案。
    def chat(self, model: str, system_prompt: str, user_prompt: str) -> str:
        # Ollama chat 接口接收 messages，格式类似 OpenAI Chat Completions。
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
        # 请求失败时抛异常。
        response.raise_for_status()
        # 返回大模型生成的文本内容。
        return response.json()["message"]["content"]
