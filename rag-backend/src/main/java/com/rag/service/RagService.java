package com.rag.service; // Service 层包名，负责业务逻辑和外部服务调用。

import com.rag.model.*; // 引入 ChatRequest、ChatResponse、DocumentInfo、SourceChunk 等 DTO。
import org.springframework.beans.factory.annotation.Value; // 读取 application.yml 里的配置。
import org.springframework.core.io.ByteArrayResource; // 把上传文件字节包装成可发送的资源。
import org.springframework.http.*; // HTTP 请求头、请求体、ContentType 等。
import org.springframework.stereotype.Service; // 声明这是 Spring Service 组件。
import org.springframework.util.LinkedMultiValueMap; // 构造 multipart/form-data 请求体。
import org.springframework.util.MultiValueMap; // multipart 请求体的接口类型。
import org.springframework.web.client.RestTemplate; // Java 调 Python HTTP 接口的客户端。
import org.springframework.web.multipart.MultipartFile; // Spring 接收前端上传文件的对象。

import java.io.IOException; // 读取上传文件字节时可能抛出异常。
import java.util.Arrays; // 把数组转 stream。
import java.util.List; // 返回列表。
import java.util.Map; // Python 返回 JSON 后，先用 Map 接收。

@Service // 交给 Spring 管理，Controller 可以自动注入这个类。
public class RagService {

    private final RestTemplate restTemplate; // HTTP 客户端，负责向 Python 服务发请求。
    private final String pythonUrl; // Python RAG 服务地址，例如 http://127.0.0.1:8001。

    // @Value 会读取 application.yml 中的 rag.python-service-url。
    // 这就是 Java 和 Python 的连接点：Java 知道 Python 服务在哪个地址。
    public RagService(@Value("${rag.python-service-url}") String pythonUrl) {
        this.restTemplate = new RestTemplate(); // 创建一个同步 HTTP 客户端。
        this.pythonUrl = pythonUrl; // 保存 Python 服务地址，后面拼接 /api/xxx。
    }

    // 健康检查：Java 调 Python 的 /api/health。
    public Map<String, Object> health() {
        // getForObject 会发 GET 请求，并把 JSON 响应反序列化成 Map。
        return restTemplate.getForObject(pythonUrl + "/api/health", Map.class);
    }

    // 上传文档：前端 -> Java MultipartFile -> Python multipart/form-data。
    public DocumentInfo uploadDocument(MultipartFile file) throws IOException {
        // 创建 HTTP 请求头。
        HttpHeaders headers = new HttpHeaders();
        // 告诉 Python：这次请求是 multipart/form-data，也就是文件上传格式。
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // 包装文件，使用原始文件名
        ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                // ByteArrayResource 默认没有文件名；Python 解析上传文件需要 filename。
                return file.getOriginalFilename();
            }
        };

        // MultiValueMap 用来构造 multipart 表单。
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        // 字段名必须叫 file，因为 Python 接口定义的是 file: UploadFile。
        body.add("file", resource);

        // 把请求体和请求头组合成一个完整 HTTP 请求实体。
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        // Python 服务返回的字段名是 snake_case，用 Map 接收再转换
        Map<String, Object> result = restTemplate.postForObject(
                pythonUrl + "/api/documents", requestEntity, Map.class);

        // 如果 Python 没返回结果，说明 Python 服务异常或网络调用失败。
        if (result == null) throw new RuntimeException("Python 服务未返回结果");

        // Python 返回 document_id/file_name/chunk_count。
        // Java record 使用 documentId/fileName/chunkCount，所以这里手动转换。
        return new DocumentInfo(
                (String) result.get("document_id"),
                (String) result.get("file_name"),
                ((Number) result.get("chunk_count")).intValue()
        );
    }

    // 查询文档列表：Java GET Python /api/documents。
    public List<DocumentInfo> listDocuments() {
        // Python 返回 snake_case 数组，转为 Java record 列表
        Map<String, Object>[] raw = restTemplate.getForObject(
                pythonUrl + "/api/documents", Map[].class);

        // Python 没返回时，给前端一个空列表，避免空指针。
        if (raw == null) return List.of();

        // 把 Map[] 转成 List<DocumentInfo>。
        return Arrays.stream(raw).map(m -> new DocumentInfo(
                (String) m.get("document_id"),
                (String) m.get("file_name"),
                ((Number) m.get("chunk_count")).intValue()
        )).toList();
    }

    // 删除文档：Java DELETE Python /api/documents/{documentId}。
    public Map<String, Object> deleteDocument(String documentId) {
        // delete 方法没有返回体，所以这里调用完后自己返回 success。
        restTemplate.delete(pythonUrl + "/api/documents/" + documentId);
        return Map.of("success", true);
    }

    // 问答接口：前端问题 -> Java -> Python RAG -> Ollama -> Java -> 前端。
    public ChatResponse chat(ChatRequest request) {
        // 构造发给 Python 服务的请求体（snake_case）
        Map<String, Object> pythonRequest = Map.of(
                "question", request.question(),
                "top_k", request.topK()
        );

        // 创建 JSON 请求头。
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // 把 JSON 请求体和请求头封装成 HttpEntity。
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(pythonRequest, headers);

        // POST 到 Python /api/chat，Python 会完成向量检索和大模型回答。
        Map<String, Object> result = restTemplate.postForObject(
                pythonUrl + "/api/chat", entity, Map.class);

        // Python 服务没返回时，直接抛异常，前端会看到请求失败。
        if (result == null) throw new RuntimeException("Python 服务未返回结果");

        // 解析 sources 数组
        List<Map<String, Object>> rawSources = (List<Map<String, Object>>) result.get("sources");
        // Python 返回的 sources 是 List<Map>，Java 需要转换成 List<SourceChunk>。
        List<SourceChunk> sources = rawSources != null ? rawSources.stream().map(s -> new SourceChunk(
                (String) s.get("document_name"),
                ((Number) s.get("chunk_index")).intValue(),
                ((Number) s.get("score")).doubleValue(),
                (String) s.get("content")
        )).toList() : List.of();

        // answer 是大模型最终回答；sources 是检索到的来源片段。
        return new ChatResponse((String) result.get("answer"), sources);
    }
}
