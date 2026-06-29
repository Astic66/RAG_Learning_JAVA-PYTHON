package com.rag.model; // DTO/模型对象所在包。

// 前端发送给 Java /api/chat 的请求体。
// record 是 Java 16+ 的简洁数据类写法，会自动生成构造器、getter、equals、hashCode。
public record ChatRequest(String question, int topK) {
    // 紧凑构造器：创建 ChatRequest 时自动执行这里的校验逻辑。
    public ChatRequest {
        // 最少召回 1 个片段，避免 topK=0 导致没有上下文。
        if (topK < 1) topK = 1;
        // 最多召回 10 个片段，避免上下文太长、模型响应太慢。
        if (topK > 10) topK = 10;
    }
}
