// =============================================================================
// RAG 企业内部知识问答系统 - 前端交互脚本
// =============================================================================
// 本文件负责：
//   1. 与后端 API 通信（健康检查、上传文档、列出文档、删除索引、提问）。
//   2. 管理对话状态（messages 数组）并持久化到 localStorage。
//   3. 渲染用户气泡、AI 气泡、加载状态、来源折叠卡片。
//   4. 处理用户交互（点击上传、点击提问、回车发送、清空对话）。
//
// 没有使用 Vue/React 等框架，全部使用原生 JavaScript 和 DOM API。
// =============================================================================

// ---------------------------------------------------------------------------
// DOM 元素引用
// ---------------------------------------------------------------------------
// 通过 querySelector 获取页面元素，方便后续操作
const healthEl = document.querySelector("#health");           // 后端状态显示
const fileInput = document.querySelector("#fileInput");       // 文件选择输入框
const uploadBtn = document.querySelector("#uploadBtn");       // 上传按钮
const docList = document.querySelector("#docList");           // 文档列表容器
const questionInput = document.querySelector("#questionInput"); // 问题输入框
const askBtn = document.querySelector("#askBtn");             // 提问按钮
const chatMessages = document.querySelector("#chatMessages"); // 聊天消息容器
const clearChatBtn = document.querySelector("#clearChatBtn"); // 清空对话按钮

// ---------------------------------------------------------------------------
// 状态管理
// ---------------------------------------------------------------------------
// messages 数组保存完整对话历史。每个元素结构为：
//   { role: "user", content: "..." }
//   { role: "assistant", content: "...", sources: [...] }
//   { role: "loading", content: "" }
let messages = loadMessages();


// ---------------------------------------------------------------------------
// 通用工具函数
// ---------------------------------------------------------------------------
async function requestJson(url, options = {}) {
  """封装 fetch 请求，统一处理 JSON 解析和错误抛出。"""
  const response = await fetch(url, options);

  // 尝试解析 JSON，如果解析失败则返回空对象
  const data = await response.json().catch(() => ({}));

  // 如果响应状态码不是 2xx，抛出后端返回的错误信息或默认提示
  if (!response.ok) {
    throw new Error(data.detail || "请求失败");
  }

  return data;
}


function escapeHtml(text) {
  """转义 HTML 特殊字符，防止 XSS 攻击。

  例如把 < 转成 &lt;，把 > 转成 &gt;。
  这里用创建一个临时 div 的方式，利用浏览器自带的 HTML 转义能力。
  """
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}


// ---------------------------------------------------------------------------
// 后端状态检查
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    // 调用 /api/health 获取后端模型信息
    const data = await requestJson("/api/health");

    // 显示已连接及当前使用的模型名称
    healthEl.textContent = `已连接：${data.chat_model} / ${data.embedding_model}`;
    healthEl.classList.add("ok");
  } catch (error) {
    // 如果请求失败，说明后端未启动或网络不通
    healthEl.textContent = "后端未连接";
  }
}


// ---------------------------------------------------------------------------
// 文档列表管理
// ---------------------------------------------------------------------------
async function loadDocuments() {
  """从后端获取已上传文档列表并渲染到左侧面板。"""
  const documents = await requestJson("/api/documents");

  // 如果没有文档，显示空状态提示
  if (!documents.length) {
    docList.innerHTML = '<p class="empty">还没有上传文档。</p>';
    return;
  }

  // 把每个文档渲染成卡片，包含文件名、块数和删除按钮
  docList.innerHTML = documents.map((doc) => `
    <article class="doc-card">
      <strong>${doc.file_name}</strong>
      <small>${doc.chunk_count} 个文本块</small>
      <button data-id="${doc.document_id}">删除索引</button>
    </article>
  `).join("");
}


async function uploadDocument() {
  """上传用户选择的文档到后端。"""
  const file = fileInput.files[0];

  // 如果没有选择文件，提示用户
  if (!file) {
    alert("先选择一个文档");
    return;
  }

  // 禁用按钮并显示处理中状态，防止重复提交
  uploadBtn.disabled = true;
  uploadBtn.textContent = "正在切块并向量化...";

  try {
    const formData = new FormData();
    formData.append("file", file);  // 把文件附加到 FormData

    // 发送 POST 请求到 /api/documents
    await requestJson("/api/documents", {
      method: "POST",
      body: formData,
    });

    // 清空文件选择框
    fileInput.value = "";

    // 刷新文档列表
    await loadDocuments();
  } catch (error) {
    alert(error.message);
  } finally {
    // 恢复按钮状态
    uploadBtn.disabled = false;
    uploadBtn.textContent = "上传并建立索引";
  }
}


// ---------------------------------------------------------------------------
// 对话渲染
// ---------------------------------------------------------------------------
function renderMessages() {
  """根据 messages 数组重新渲染整个聊天区域。"""

  // 如果没有任何消息，显示欢迎提示
  if (!messages.length) {
    chatMessages.innerHTML = '<p class="welcome-tip">先上传一份企业制度、产品说明或课程笔记，然后开始提问。</p>';
    return;
  }

  // 遍历 messages，根据 role 渲染不同类型的气泡
  chatMessages.innerHTML = messages.map((msg, index) => {
    // 用户消息：右对齐，使用 user-bubble 样式
    if (msg.role === "user") {
      return `
        <div class="message user-message" data-index="${index}">
          <div class="bubble user-bubble">${escapeHtml(msg.content)}</div>
        </div>
      `;
    }

    // AI 助手消息：左对齐，使用 assistant-bubble 样式，并可能包含来源
    if (msg.role === "assistant") {
      // 如果有来源片段，渲染成可折叠的 details 元素
      const sourcesHtml = msg.sources && msg.sources.length
        ? `<div class="sources">
            ${msg.sources.map((source, idx) => `
              <details class="source-card">
                <summary>来源 ${idx + 1}：${escapeHtml(source.document_name)} / 第 ${source.chunk_index} 块 / 相似度 ${source.score}</summary>
                <p>${escapeHtml(source.content.slice(0, 260))}${source.content.length > 260 ? "..." : ""}</p>
              </details>
            `).join("")}
           </div>`
        : "";

      return `
        <div class="message assistant-message" data-index="${index}">
          <div class="bubble assistant-bubble">
            <div class="answer-content">${escapeHtml(msg.content)}</div>
            ${sourcesHtml}
          </div>
        </div>
      `;
    }

    // 加载中消息：显示一个带有动画的提示气泡
    if (msg.role === "loading") {
      return `
        <div class="message assistant-message" data-index="${index}">
          <div class="bubble assistant-bubble loading-bubble">
            <span class="loading-dots"><i></i><i></i><i></i></span> 正在检索知识库并调用本地大模型...
          </div>
        </div>
      `;
    }

    return "";
  }).join("");

  // 渲染完成后自动滚动到底部，让用户看到最新消息
  scrollToBottom();
}


function scrollToBottom() {
  """把聊天区域滚动到最底部。"""
  chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ---------------------------------------------------------------------------
// 对话历史持久化
// ---------------------------------------------------------------------------
function saveMessages() {
  """把 messages 数组保存到浏览器的 localStorage，刷新页面后不丢失。"""
  localStorage.setItem("rag_chat_messages", JSON.stringify(messages));
}


function loadMessages() {
  """从 localStorage 加载对话历史。"""
  try {
    const raw = localStorage.getItem("rag_chat_messages");
    return raw ? JSON.parse(raw) : [];
  } catch {
    // 如果解析失败（例如数据被手动修改过），返回空数组
    return [];
  }
}


function clearMessages() {
  """清空当前对话历史和 localStorage 中的记录。"""
  messages = [];
  saveMessages();
  renderMessages();
}


// ---------------------------------------------------------------------------
// 提问逻辑
// ---------------------------------------------------------------------------
async function askQuestion() {
  """处理用户提问：添加用户气泡、显示加载状态、调用后端、渲染 AI 回答。"""
  const question = questionInput.value.trim();

  if (!question) {
    alert("先输入问题");
    return;
  }

  // 1. 先把用户问题加入消息列表并立即渲染
  messages.push({ role: "user", content: question });

  // 2. 显示加载中气泡
  messages.push({ role: "loading", content: "" });
  renderMessages();

  // 3. 清空输入框
  questionInput.value = "";

  // 4. 禁用提问按钮，防止重复提交
  askBtn.disabled = true;
  askBtn.textContent = "思考中...";

  try {
    // 5. 调用后端问答接口
    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, top_k: 4}),
    });

    // 6. 移除加载气泡
    messages.pop();

    // 7. 把 AI 回答加入消息列表
    messages.push({
      role: "assistant",
      content: data.answer,
      sources: data.sources || [],
    });
  } catch (error) {
    // 请求失败时，移除加载气泡并显示错误信息
    messages.pop();
    messages.push({
      role: "assistant",
      content: `请求出错：${error.message}`,
      sources: [],
      isError: true,
    });
  } finally {
    // 保存对话历史并重新渲染
    saveMessages();
    renderMessages();

    // 恢复提问按钮
    askBtn.disabled = false;
    askBtn.textContent = "提问";
  }
}


// ---------------------------------------------------------------------------
// 事件监听
// ---------------------------------------------------------------------------
// 文档列表点击事件：点击删除按钮时删除对应文档索引
docList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) {
    return;  // 如果点击的不是删除按钮，直接忽略
  }

  // 调用 DELETE /api/documents/{id}
  await requestJson(`/api/documents/${button.dataset.id}`, {method: "DELETE"});

  // 刷新文档列表
  await loadDocuments();
});

// 上传按钮点击事件
uploadBtn.addEventListener("click", uploadDocument);

// 提问按钮点击事件
askBtn.addEventListener("click", askQuestion);

// 清空对话按钮点击事件
clearChatBtn.addEventListener("click", clearMessages);

// 输入框键盘事件：按回车发送（Shift+Enter 换行）
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();  // 阻止默认换行行为
    askQuestion();
  }
});


// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------
checkHealth();        // 页面加载时检查后端连接
loadDocuments();      // 页面加载时获取文档列表
renderMessages();     // 页面加载时渲染历史对话（如果有）
