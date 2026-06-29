package com.rag.controller; // Controller 层包名，负责文档相关 HTTP 接口。

import com.rag.model.DocumentInfo; // 文档信息响应对象。
import com.rag.model.HealthResponse; // 健康检查响应对象。
import com.rag.service.RagService; // 业务服务，负责调用 Python RAG 服务。
import org.springframework.http.ResponseEntity; // HTTP 响应包装对象。
import org.springframework.web.bind.annotation.*; // REST 接口注解。
import org.springframework.web.multipart.MultipartFile; // Spring 接收上传文件的类型。

import java.io.IOException; // 上传文件读取可能抛出的异常。
import java.util.List; // 返回文档列表。
import java.util.Map; // 返回删除结果等简单结构。

@RestController // 当前类提供 JSON API。
@RequestMapping("/api") // 所有接口统一挂在 /api 下。
public class DocumentController {

    private final RagService ragService; // Java 与 Python RAG 服务之间的中间层。

    // 构造器注入，Spring 自动创建并传入 RagService。
    public DocumentController(RagService ragService) {
        this.ragService = ragService;
    }

    @GetMapping("/health") // 前端启动时会调用这个接口检查服务是否正常。
    public ResponseEntity<HealthResponse> health() {
        // 调用 Python /api/health，确认 Python 服务和模型配置。
        Map<String, Object> result = ragService.health();
        // Python 返回 snake_case 字段；Java 返回给前端时转成 camelCase record。
        return ResponseEntity.ok(new HealthResponse(
                (String) result.get("status"),
                (String) result.get("chat_model"),
                (String) result.get("embedding_model")
        ));
    }

    @PostMapping("/documents") // 前端上传文档会调用这个接口。
    public ResponseEntity<DocumentInfo> uploadDocument(@RequestParam("file") MultipartFile file) throws IOException {
        // Java 收到 MultipartFile 后，再转发给 Python RAG 服务做解析、切块、向量化。
        return ResponseEntity.ok(ragService.uploadDocument(file));
    }

    @GetMapping("/documents") // 获取已经入库的文档列表。
    public ResponseEntity<List<DocumentInfo>> listDocuments() {
        // 实际文档索引保存在 Python 服务的 JSON 向量库里，所以这里继续转发给 Python。
        return ResponseEntity.ok(ragService.listDocuments());
    }

    @DeleteMapping("/documents/{documentId}") // 删除某个文档对应的所有向量块。
    public ResponseEntity<Map<String, Object>> deleteDocument(@PathVariable String documentId) {
        // documentId 来自 URL 路径，比如 /api/documents/xxx。
        return ResponseEntity.ok(ragService.deleteDocument(documentId));
    }
}
