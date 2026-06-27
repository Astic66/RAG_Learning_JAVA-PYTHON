# RAG 企业内部知识问答系统后端

这是一个学习版 RAG 后端，使用 FastAPI + Ollama 本地模型实现。

## RAG 流程

```text
上传文档
-> 解析文本
-> 文本切块
-> nomic-embed-text 生成向量
-> JSON 向量库存储
-> 用户提问
-> 问题向量化
-> 余弦相似度检索
-> qwen2.5:7b 基于资料回答
```

## 启动

```bash
cd /d D:\AI相关\RAG项目\RAG企业内部知识问答系统\backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```
