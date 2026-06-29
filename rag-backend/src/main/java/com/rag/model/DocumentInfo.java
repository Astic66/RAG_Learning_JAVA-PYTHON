package com.rag.model; // DTO/模型对象所在包。

// 文档列表/上传结果返回对象。
// documentId：一次上传生成的文档唯一 ID。
// fileName：原始文件名。
// chunkCount：这个文档被切成了多少个文本块。
public record DocumentInfo(String documentId, String fileName, int chunkCount) {}
