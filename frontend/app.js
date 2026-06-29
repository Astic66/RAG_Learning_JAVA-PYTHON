const healthEl = document.querySelector("#health");
const fileInput = document.querySelector("#fileInput");
const uploadBtn = document.querySelector("#uploadBtn");
const docList = document.querySelector("#docList");
const questionInput = document.querySelector("#questionInput");
const askBtn = document.querySelector("#askBtn");
const chatMessages = document.querySelector("#chatMessages");
const clearChatBtn = document.querySelector("#clearChatBtn");

let messages = loadMessages();

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.message || "请求失败");
  }
  return data;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function checkHealth() {
  try {
    const data = await requestJson("/api/health");
    healthEl.textContent = "已连接：" + data.chatModel + " / " + data.embeddingModel;
    healthEl.classList.add("ok");
  } catch (error) {
    healthEl.textContent = "后端未连接";
  }
}

async function loadDocuments() {
  const documents = await requestJson("/api/documents");
  if (!documents.length) {
    docList.innerHTML = '<p class="empty">还没有上传文档。</p>';
    return;
  }
  docList.innerHTML = documents.map((doc) =>
    '<article class="doc-card">' +
      '<strong>' + escapeHtml(doc.fileName) + '</strong>' +
      '<small>' + doc.chunkCount + ' 个文本块</small>' +
      '<button data-id="' + doc.documentId + '">删除索引</button>' +
    '</article>'
  ).join("");
}

async function uploadDocument() {
  const file = fileInput.files[0];
  if (!file) {
    alert("先选择一个文档");
    return;
  }
  uploadBtn.disabled = true;
  uploadBtn.textContent = "正在切块并向量化...";
  try {
    const formData = new FormData();
    formData.append("file", file);
    await requestJson("/api/documents", {
      method: "POST",
      body: formData,
    });
    fileInput.value = "";
    await loadDocuments();
  } catch (error) {
    alert(error.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "上传并建立索引";
  }
}

function renderMessages() {
  if (!messages.length) {
    chatMessages.innerHTML = '<p class="welcome-tip">先上传一份企业制度、产品说明或课程笔记，然后开始提问。</p>';
    return;
  }
  chatMessages.innerHTML = messages.map((msg, index) => {
    if (msg.role === "user") {
      return '<div class="message user-message" data-index="' + index + '">' +
               '<div class="bubble user-bubble">' + escapeHtml(msg.content) + '</div>' +
             '</div>';
    }
    if (msg.role === "assistant") {
      const sourcesHtml = msg.sources && msg.sources.length
        ? '<div class="sources">' +
            msg.sources.map((source, idx) =>
              '<details class="source-card">' +
                '<summary>来源 ' + (idx + 1) + '：' + escapeHtml(source.documentName) + ' / 第 ' + source.chunkIndex + ' 块 / 相似度 ' + source.score + '</summary>' +
                '<p>' + escapeHtml(source.content.slice(0, 260)) + (source.content.length > 260 ? "..." : "") + '</p>' +
              '</details>'
            ).join("") +
          '</div>'
        : "";
      return '<div class="message assistant-message" data-index="' + index + '">' +
               '<div class="bubble assistant-bubble">' +
                 '<div class="answer-content">' + escapeHtml(msg.content) + '</div>' +
                 sourcesHtml +
               '</div>' +
             '</div>';
    }
    if (msg.role === "loading") {
      return '<div class="message assistant-message" data-index="' + index + '">' +
               '<div class="bubble assistant-bubble loading-bubble">' +
                 '<span class="loading-dots"><i></i><i></i><i></i></span> 正在检索知识库并调用本地大模型...' +
               '</div>' +
             '</div>';
    }
    return "";
  }).join("");
  scrollToBottom();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function saveMessages() {
  localStorage.setItem("rag_chat_messages", JSON.stringify(messages));
}

function loadMessages() {
  try {
    const raw = localStorage.getItem("rag_chat_messages");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function clearMessages() {
  messages = [];
  saveMessages();
  renderMessages();
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    alert("先输入问题");
    return;
  }
  messages.push({ role: "user", content: question });
  messages.push({ role: "loading", content: "" });
  renderMessages();
  questionInput.value = "";
  askBtn.disabled = true;
  askBtn.textContent = "思考中...";

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, topK: 4 }),
    });
    messages.pop();
    messages.push({
      role: "assistant",
      content: data.answer,
      sources: data.sources || [],
    });
  } catch (error) {
    messages.pop();
    messages.push({
      role: "assistant",
      content: "请求出错：" + error.message,
      sources: [],
      isError: true,
    });
  } finally {
    saveMessages();
    renderMessages();
    askBtn.disabled = false;
    askBtn.textContent = "提问";
  }
}

docList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) return;
  await requestJson("/api/documents/" + button.dataset.id, { method: "DELETE" });
  await loadDocuments();
});

uploadBtn.addEventListener("click", uploadDocument);
askBtn.addEventListener("click", askQuestion);
clearChatBtn.addEventListener("click", clearMessages);

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});

checkHealth();
loadDocuments();
renderMessages();
