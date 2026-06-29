package com.rag.model; // DTO/模型对象所在包。

// 一条检索来源片段。
// documentName：来自哪个文档。
// chunkIndex：来自文档的第几个切块。
// score：问题向量和文本块向量的相似度。
// content：文本块原文。
public record SourceChunk(String documentName, int chunkIndex, double score, String content) {}
