package com.rag.controller; // Controller 层包名，负责接收浏览器 HTTP 请求。

import com.rag.model.ChatRequest; // 前端提问请求体。
import com.rag.model.ChatResponse; // 返回给前端的问答结果。
import com.rag.service.RagService; // 业务服务，内部会转发请求给 Python RAG 服务。
import org.springframework.http.ResponseEntity; // HTTP 响应包装对象。
import org.springframework.web.bind.annotation.*; // RestController、PostMapping、RequestBody 等注解。

@RestController // 表示这个类返回 JSON 数据，不返回 JSP/模板页面。
@RequestMapping("/api") // 这个 Controller 下所有接口都以 /api 开头。
public class ChatController {

    private final RagService ragService; // Java 业务层对象，用来调用 Python 服务。

    // 构造器注入。Spring 会自动把 RagService 实例传进来。
    public ChatController(RagService ragService) {
        this.ragService = ragService;
    }

    @PostMapping("/chat") // 对应前端 fetch("/api/chat")。
    public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest request) {
        // Java 本身不做向量检索，它把请求交给 RagService，再由 RagService 调 Python。
        return ResponseEntity.ok(ragService.chat(request));
    }
}
