package com.rag.model; // DTO/模型对象所在包。

import java.util.List; // sources 是多个来源片段，所以用 List。

// Java /api/chat 返回给前端的数据结构。
// answer：大模型最终回答。
// sources：RAG 检索出来的资料来源，方便前端展示“答案依据”。
public record ChatResponse(String answer, List<SourceChunk> sources) {}
