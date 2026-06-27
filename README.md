# RAG 企业内部知识问答系统

这是一个适合学习的本地 RAG 项目，使用：

- 后端：Python 标准库 HTTP Server
- 前端：原生 HTML/CSS/JS
- 聊天模型：Ollama `qwen2.5:7b`
- 向量模型：Ollama `nomic-embed-text`
- 向量库：学习用 JSON 本地向量库

## 先确认模型

```bash
ollama list
```

需要看到：

```text
qwen2.5:7b
nomic-embed-text
```

没有的话执行：

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## 启动

双击：

```text
启动后端.bat
```

然后浏览器打开：

```text
http://127.0.0.1:8000
```

## 学习重点

代码最值得看的几个文件：

- `backend/app/main.py`：接口入口，串起完整 RAG 流程
- `backend/app/services/document_loader.py`：文档解析
- `backend/app/services/text_splitter.py`：文本切块
- `backend/app/services/ollama_client.py`：调用 Ollama
- `backend/app/services/vector_store.py`：向量存储和相似度检索

## 当前版本能力

- 上传 `txt/md`
- 自动切块
- 调用本地 embedding 模型向量化
- 保存本地向量索引
- 用户提问
- 检索相关片段
- 调用本地大模型生成答案
- 展示答案来源

说明：这个默认版本是零第三方依赖，主要为了避开 Python 3.14 依赖编译问题。PDF/DOCX 解析需要额外库，后面建议装 Python 3.11/3.12 后再加。
