package com.rag.model; // DTO/模型对象所在包。

// 健康检查返回对象。
// status：服务状态。
// chatModel：聊天模型，例如 qwen2.5:7b。
// embeddingModel：向量模型，例如 nomic-embed-text。
public record HealthResponse(String status, String chatModel, String embeddingModel) {}
