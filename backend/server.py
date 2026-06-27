# =============================================================================
# RAG 企业内部知识问答系统 - 零依赖标准库后端入口
# =============================================================================
# 设计目标：
#   1. 不依赖任何第三方 Python 包，避免 Python 3.14 等环境下 wheels 编译问题。
#   2. 用标准库 http.server 提供 REST API 并同时服务前端静态文件。
#   3. 完整演示 RAG 的“上传入库”和“提问生成”两条链路。
# =============================================================================

# ---------------------------------------------------------------------------
# 标准库导入
# ---------------------------------------------------------------------------
import json                         # 序列化/反序列化向量库（JSON 文件）
import mimetypes                    # 根据文件后缀推断 Content-Type，用于返回静态资源
import shutil                       # 文件流拷贝，用于返回前端静态文件
import uuid                         # 生成全局唯一标识：文档 ID、块 ID
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # HTTP 服务基类与多线程服务器
from pathlib import Path            # 面向对象的路径操作，自动处理 Windows/Linux 路径差异
from urllib.parse import unquote, urlparse  # URL 路径解析与中文解码
from urllib.request import Request, urlopen  # 标准库 HTTP 客户端，用于调用 Ollama


# ---------------------------------------------------------------------------
# 路径与目录配置
# ---------------------------------------------------------------------------
# BASE_DIR 指向 backend 目录，PROJECT_DIR 指向项目根目录。
# 这样无论脚本从哪里被启动，路径都能正确解析。
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"          # 前端静态资源目录
STORAGE_DIR = BASE_DIR / "storage"               # 数据持久化根目录
DOCUMENTS_DIR = STORAGE_DIR / "documents"        # 上传的原始文档保存位置
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.json"  # 向量库文件路径


# ---------------------------------------------------------------------------
# 模型与分块参数配置
# ---------------------------------------------------------------------------
# 这些常量集中在文件顶部，方便学习者一次性看懂并修改。
OLLAMA_BASE_URL = "http://127.0.0.1:11434"     # Ollama 本地服务地址
CHAT_MODEL = "qwen2.5:7b"                      # 负责生成答案的聊天模型
EMBEDDING_MODEL = "nomic-embed-text"           # 负责把文本转成向量的嵌入模型
CHUNK_SIZE = 800                               # 每个文本块最大字符数
CHUNK_OVERLAP = 120                            # 相邻块之间的重叠字符数，保留上下文


# ---------------------------------------------------------------------------
# 运行时自动创建必要目录
# ---------------------------------------------------------------------------
# 如果目录不存在，mkdir(parents=True, exist_ok=True) 会递归创建，避免启动时报错。
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 向量库读写
# ---------------------------------------------------------------------------
def load_store() -> list[dict]:
    """从 JSON 文件加载所有向量块。

    如果向量库文件不存在（首次运行），返回空列表。
    每个元素都是一个字典，包含 id/document_id/document_name/chunk_index/content/embedding。
    """
    if not VECTOR_STORE_PATH.exists():           # 检查文件是否存在
        return []                                # 不存在则返回空列表
    return json.loads(VECTOR_STORE_PATH.read_text(encoding="utf-8"))  # 读取并解析 JSON


def save_store(items: list[dict]) -> None:
    """把所有向量块写回 JSON 文件。

    ensure_ascii=False 保证中文正常显示，indent=2 让文件可读，便于学习时打开查看。
    """
    VECTOR_STORE_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),  # 格式化为可读 JSON
        encoding="utf-8",                                 # 使用 UTF-8 编码
    )


# ---------------------------------------------------------------------------
# 文本分块
# ---------------------------------------------------------------------------
def split_text(text: str) -> list[str]:
    """文本切块：RAG 入库前必须把长文档切成较小片段。

    实现细节：
    1. 先按行清洗，去掉空行并把每行首尾空白去掉。
    2. 用滑动窗口遍历文本，窗口大小 CHUNK_SIZE。
    3. 每次前进 CHUNK_SIZE - CHUNK_OVERLAP 个字符，保证相邻块有重叠。
       这样可以避免一句话刚好被从中间切断，提高后续检索的召回率。
    """
    # 清洗文本：去掉空行，去掉每行首尾空格，再用换行符重新拼接
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    chunks = []              # 用于收集切出来的文本块
    start = 0                # 当前窗口的起始位置

    while start < len(cleaned):                     # 当窗口起点未超过文本末尾时继续
        end = min(start + CHUNK_SIZE, len(cleaned)) # 窗口终点，不超过文本末尾
        chunk = cleaned[start:end].strip()          # 取出当前窗口内容并去首尾空

        if chunk:                                    # 如果内容非空才加入结果
            chunks.append(chunk)

        if end == len(cleaned):                      # 如果已经到达文本末尾，结束循环
            break

        # 下一个窗口起点：当前终点往前回退 CHUNK_SIZE 个字符，保证重叠。
        # max(..., start + 1) 防止死循环（当 overlap 过大时至少前进 1 个字符）。
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# Ollama HTTP 调用
# ---------------------------------------------------------------------------
def ollama_post(path: str, payload: dict, timeout: int = 180) -> dict:
    """向本地 Ollama 服务发送 POST 请求。

    这是一个通用封装， embedding 和 chat 都会调用它。
    使用标准库 urllib.request，因此不需要 requests 等第三方库。
    """
    body = json.dumps(payload).encode("utf-8")       # 把字典序列化为 JSON 字节流

    request = Request(
        f"{OLLAMA_BASE_URL}{path}",                  # 拼接完整 URL，例如 http://127.0.0.1:11434/api/embed
        data=body,                                    # 请求体
        headers={"Content-Type": "application/json"}, # 声明发送的是 JSON
        method="POST",                                # HTTP 方法
    )

    with urlopen(request, timeout=timeout) as response:       # 发送请求并获取响应
        return json.loads(response.read().decode("utf-8"))    # 读取响应体并解析 JSON


def embed_text(text: str) -> list[float]:
    """调用本地 embedding 模型，把文本变成向量。

    Ollama 的 /api/embed 接口返回结构为：
        {"embeddings": [[0.1, -0.2, ...], ...]}
    因为只传入一个文本，所以取第 0 个 embedding。
    """
    data = ollama_post("/api/embed", {"model": EMBEDDING_MODEL, "input": text})
    return data["embeddings"][0]                      # 返回第一个（也是唯一一个）向量


def chat_with_context(question: str, context: str) -> str:
    """调用本地聊天模型，让它基于检索上下文回答。

    这里把“系统提示词”和“用户提示词”一起发给 Ollama，
    明确告诉模型必须依据资料回答，减少幻觉。
    """
    system_prompt = (
        "你是企业内部知识库问答助手。"
        "你必须优先依据给定资料回答问题。"
        "如果资料中没有答案，请直接说资料中没有找到相关信息，不要编造。"
        "回答要清晰、简洁，并使用中文。"
    )

    # 把用户问题和检索到的上下文拼接成用户提示词
    user_prompt = f"用户问题：{question}\n\n可用资料：\n{context}"

    data = ollama_post(
        "/api/chat",                                  # Ollama 聊天接口
        {
            "model": CHAT_MODEL,                      # 指定聊天模型
            "stream": False,                          # 非流式，一次性返回完整回答
            "messages": [                             # OpenAI 风格的消息列表
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )

    return data["message"]["content"]                 # 提取模型生成的文本


# ---------------------------------------------------------------------------
# 向量相似度计算
# ---------------------------------------------------------------------------
def vector_norm(vector: list[float]) -> float:
    """计算向量的 L2 范数（模长）。

    公式：||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
    """
    return sum(value * value for value in vector) ** 0.5


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量之间的余弦相似度。

    公式：cos(A, B) = (A·B) / (||A|| × ||B||)
    结果范围 [-1, 1]，对于非负 embedding 通常落在 [0, 1]。
    值越大表示两段文本语义越接近。
    """
    denominator = vector_norm(left) * vector_norm(right)  # 分母：两个向量模长乘积

    if denominator == 0:                                  # 防止除零
        return 0.0

    # 分子：两个向量的点积
    return sum(a * b for a, b in zip(left, right)) / denominator


def search_chunks(question: str, top_k: int = 4) -> list[dict]:
    """问题向量化后，在本地 JSON 向量库中做相似度检索。

    这是 RAG 的核心检索步骤：
    1. 把用户问题转成向量。
    2. 用余弦相似度与库中每个块比较。
    3. 按相似度降序排序，取前 top_k 个。
    """
    store = load_store()                              # 加载所有向量块
    query_embedding = embed_text(question)            # 把问题转成向量

    scored = []                                       # 保存 (相似度, 块) 二元组
    for item in store:                                # 遍历所有块
        score = cosine_similarity(query_embedding, item["embedding"])  # 计算相似度
        scored.append((score, item))                  # 加入待排序列表

    scored.sort(key=lambda pair: pair[0], reverse=True)  # 按相似度从高到低排序

    results = []
    for score, item in scored[:top_k]:                # 只取前 top_k 个
        copied = dict(item)                           # 复制一份，避免修改原始数据
        copied["score"] = round(score, 4)             # 保留 4 位小数，方便展示
        results.append(copied)

    return results


# ---------------------------------------------------------------------------
# 文件上传解析（手写 multipart/form-data 解析器）
# ---------------------------------------------------------------------------
def parse_multipart_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    """解析浏览器 FormData 上传的单个文件。

    这是学习项目的简化版 multipart 解析，只处理一个 file 字段。
    真实生产环境建议用 FastAPI/Flask 等框架自带的文件处理。
    """
    marker = "boundary="                               # multipart boundary 的标记

    if marker not in content_type:                     # 如果 Content-Type 里没有 boundary，说明不是 multipart
        raise ValueError("缺少 multipart boundary")

    # boundary 前面要加 "--"，例如 boundary=----WebKitFormBoundary...
    boundary = ("--" + content_type.split(marker, 1)[1]).encode("utf-8")
    parts = body.split(boundary)                       # 用 boundary 把请求体切成多段

    for part in parts:                                 # 遍历每一段
        # 只处理包含 Content-Disposition 且带有 filename 的段，那才是真正上传的文件
        if b'Content-Disposition:' not in part or b'filename="' not in part:
            continue

        # 用 \r\n\r\n 把 HTTP 头和文件内容分开
        header, file_content = part.split(b"\r\n\r\n", 1)
        header_text = header.decode("utf-8", errors="ignore")  # 头部是文本，可安全解码

        # 从 Content-Disposition 头中提取 filename="..."
        filename = header_text.split('filename="', 1)[1].split('"', 1)[0]

        # 去掉文件内容末尾可能的 \r\n- 残留字符
        file_content = file_content.rstrip(b"\r\n-")

        return filename, file_content                  # 返回文件名和二进制内容

    raise ValueError("没有找到上传文件")                 # 遍历完都没找到文件则报错


# ---------------------------------------------------------------------------
# HTTP 请求处理器
# ---------------------------------------------------------------------------
class RagHandler(BaseHTTPRequestHandler):
    """自定义 HTTP 请求处理器。

    继承自 BaseHTTPRequestHandler，分别实现：
    - do_OPTIONS：处理跨域预检请求
    - do_GET：健康检查、文档列表、静态资源
    - do_POST：上传文档、提问
    - do_DELETE：删除文档索引
    """

    def _send_json(self, data: dict | list, status: int = 200) -> None:
        """统一封装 JSON 响应，处理编码、跨域头和 Content-Length。"""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")  # JSON 序列化并编码

        self.send_response(status)                        # 发送 HTTP 状态码
        self.send_header("Content-Type", "application/json; charset=utf-8")  # 声明 JSON + UTF-8
        self.send_header("Access-Control-Allow-Origin", "*")                 # 允许跨域
        self.send_header("Content-Length", str(len(body)))                   # 告知响应体长度
        self.end_headers()                                # 结束响应头
        self.wfile.write(body)                            # 写入响应体

    def _send_error(self, message: str, status: int = 400) -> None:
        """统一返回错误信息，保持和成功响应一致的 JSON 格式。"""
        self._send_json({"detail": message}, status)

    def do_OPTIONS(self) -> None:
        """处理浏览器在 POST/DELETE 前自动发送的预检请求（CORS Preflight）。"""
        self.send_response(204)                           # 204 No Content
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """处理 GET 请求。"""
        path = urlparse(self.path).path                   # 解析 URL 路径部分

        if path == "/api/health":                         # 健康检查接口
            self._send_json(
                {
                    "status": "ok",
                    "chat_model": CHAT_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                }
            )
            return

        if path == "/api/documents":                      # 文档列表接口
            grouped = {}                                  # 按 document_id 分组

            for item in load_store():                     # 遍历所有向量块
                doc_id = item["document_id"]

                # 如果这是第一次遇到该文档，先初始化分组项
                grouped.setdefault(
                    doc_id,
                    {
                        "document_id": doc_id,
                        "file_name": item["document_name"],
                        "chunk_count": 0,
                    },
                )

                grouped[doc_id]["chunk_count"] += 1       # 累加该文档的块数量

            self._send_json(list(grouped.values()))       # 把分组字典转成列表返回
            return

        self._serve_static(path)                          # 其余请求当作静态资源处理

    def do_POST(self) -> None:
        """处理 POST 请求。"""
        path = urlparse(self.path).path                   # 解析请求路径
        content_length = int(self.headers.get("Content-Length", "0"))  # 获取请求体长度
        body = self.rfile.read(content_length)            # 读取请求体字节流

        try:
            if path == "/api/documents":                  # 上传文档接口
                self._upload_document(body)
                return

            if path == "/api/chat":                       # 提问接口
                payload = json.loads(body.decode("utf-8"))  # 把 JSON 字符串解析成字典
                self._chat(payload)
                return

            self._send_error("接口不存在", 404)            # 路径不匹配

        except Exception as exc:                          # 捕获所有异常，防止服务崩溃
            self._send_error(str(exc), 500)

    def do_DELETE(self) -> None:
        """处理 DELETE 请求，用于删除某个文档的向量索引。"""
        path = urlparse(self.path).path
        prefix = "/api/documents/"

        if not path.startswith(prefix):                   # 路径格式不对
            self._send_error("接口不存在", 404)
            return

        # 从 URL 中提取 document_id，例如 /api/documents/abc -> abc
        document_id = unquote(path[len(prefix):])

        store = load_store()                              # 加载现有向量库
        # 过滤掉属于该 document_id 的所有块
        new_store = [item for item in store if item["document_id"] != document_id]
        save_store(new_store)                             # 写回文件

        self._send_json({"deleted_chunks": len(store) - len(new_store)})  # 返回删除块数

    def _upload_document(self, body: bytes) -> None:
        """文档上传并建立向量索引。

        完整流程：
        multipart 解析 -> 保存原始文件 -> 文本解码 -> 文本分块 ->
        每个块 Embedding -> 写入 JSON 向量库
        """
        # 解析浏览器上传的文件，得到文件名和字节内容
        filename, file_content = parse_multipart_file(
            self.headers.get("Content-Type", ""),
            body,
        )

        suffix = Path(filename).suffix.lower()            # 获取文件后缀

        # 当前零依赖版本只支持 txt 和 md
        if suffix not in {".txt", ".md"}:
            raise ValueError("当前零依赖版本先支持 txt、md。PDF/DOCX 后面可以再接解析库。")

        safe_name = Path(filename).name                   # 取文件名（去掉任何路径穿越）
        saved_path = DOCUMENTS_DIR / safe_name            # 拼接保存路径
        saved_path.write_bytes(file_content)              # 把原始文件字节写入磁盘

        text = file_content.decode("utf-8", errors="ignore")  # 把字节按 UTF-8 解码成字符串
        chunks = split_text(text)                         # 调用分块函数

        if not chunks:                                    # 如果分块结果为空，说明文档没有有效内容
            raise ValueError("文档内容为空")

        document_id = str(uuid.uuid4())                   # 为该文档生成唯一 ID
        store = load_store()                              # 加载现有向量库

        for index, chunk in enumerate(chunks):            # 遍历每个文本块
            store.append(
                {
                    "id": str(uuid.uuid4()),              # 每个块也有自己的唯一 ID
                    "document_id": document_id,           # 块属于哪个文档
                    "document_name": safe_name,           # 文档原始文件名
                    "chunk_index": index,                 # 块在文档中的序号
                    "content": chunk,                     # 块的文本内容
                    "embedding": embed_text(chunk),       # 块的向量表示
                }
            )

        save_store(store)                                 # 把所有更新写回 JSON 文件

        self._send_json(
            {
                "document_id": document_id,
                "file_name": safe_name,
                "chunk_count": len(chunks),
            }
        )

    def _chat(self, payload: dict) -> None:
        """问答接口。

        完整流程：
        获取问题 -> 检索相关块 -> 拼接上下文 -> 调用 LLM 生成答案 -> 返回答案和来源
        """
        question = (payload.get("question") or "").strip()  # 获取并清洗问题
        top_k = int(payload.get("top_k") or 4)              # 默认召回 4 个片段

        if not question:                                    # 问题为空则报错
            raise ValueError("问题不能为空")

        if not load_store():                                # 向量库为空时提示先上传文档
            self._send_error("请先上传文档再提问")
            return

        results = search_chunks(question, top_k)            # 执行相似度检索

        # 把检索结果格式化成上下文字符串
        context = "\n\n".join(
            f"资料 {index + 1}：\n来源：{item['document_name']} 第 {item['chunk_index']} 块\n内容：{item['content']}"
            for index, item in enumerate(results)
        )

        answer = chat_with_context(question, context)       # 调用 LLM 生成答案

        self._send_json(
            {
                "answer": answer,                          # 模型生成的回答
                "sources": [                               # 引用的来源片段
                    {
                        "document_name": item["document_name"],
                        "chunk_index": item["chunk_index"],
                        "score": item["score"],
                        "content": item["content"],
                    }
                    for item in results
                ],
            }
        )

    def _serve_static(self, path: str) -> None:
        """返回前端静态资源文件。"""
        if path == "/":                                   # 根路径返回首页
            file_path = FRONTEND_DIR / "index.html"
        elif path.startswith("/assets/"):                 # /assets/ 路径映射到 frontend 目录
            file_path = FRONTEND_DIR / path.replace("/assets/", "", 1)
        else:                                             # 其他路径直接映射
            file_path = FRONTEND_DIR / path.lstrip("/")

        if not file_path.exists() or not file_path.is_file():  # 文件不存在则 404
            self._send_error("页面不存在", 404)
            return

        # 根据文件后缀自动推断 Content-Type，例如 .css -> text/css
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

        self.send_response(200)                           # 200 OK
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.end_headers()

        with file_path.open("rb") as file:                # 以二进制方式打开文件
            shutil.copyfileobj(file, self.wfile)          # 把文件内容拷贝到响应流


# ---------------------------------------------------------------------------
# 程序入口
# ---------------------------------------------------------------------------
def main() -> None:
    """启动多线程 HTTP 服务器。"""
    # ThreadingHTTPServer 会为每个请求开一个新线程，支持基础并发
    server = ThreadingHTTPServer(("127.0.0.1", 8000), RagHandler)
    print("RAG 企业内部知识问答系统已启动：http://127.0.0.1:8000")
    server.serve_forever()                              # 永久监听请求


if __name__ == "__main__":
    main()                                              # 直接运行脚本时启动服务
