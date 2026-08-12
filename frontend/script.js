/* ==========================================================================
   Marginalia — frontend logic
   Talks to the FastAPI backend at API_BASE.
   ========================================================================== */

const API_BASE = "/api";

const state = {
  sessionId: null,
  filename: null,
};

// ---------------------------------------------------------------------------
// Element refs
// ---------------------------------------------------------------------------

const el = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  uploadStatus: document.getElementById("uploadStatus"),
  uploadPanel: document.getElementById("uploadPanel"),
  docPanel: document.getElementById("docPanel"),
  docName: document.getElementById("docName"),
  docPages: document.getElementById("docPages"),
  docChunks: document.getElementById("docChunks"),
  removeDocBtn: document.getElementById("removeDocBtn"),
  docBadge: document.getElementById("docBadge"),
  docBadgeName: document.getElementById("docBadgeName"),

  tabNav: document.getElementById("tabNav"),
  emptyState: document.getElementById("emptyState"),

  chatLog: document.getElementById("chatLog"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatSendBtn: document.getElementById("chatSendBtn"),

  generateBlogBtn: document.getElementById("generateBlogBtn"),
  blogOutput: document.getElementById("blogOutput"),

  generateGlossaryBtn: document.getElementById("generateGlossaryBtn"),
  glossaryOutput: document.getElementById("glossaryOutput"),
};

// ---------------------------------------------------------------------------
// Upload flow
// ---------------------------------------------------------------------------

el.dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  el.dropzone.classList.add("dragover");
});
el.dropzone.addEventListener("dragleave", () =>
  el.dropzone.classList.remove("dragover")
);
el.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  el.dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

el.fileInput.addEventListener("change", () => {
  const file = el.fileInput.files[0];
  if (file) uploadFile(file);
});

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showUploadStatus("Please choose a PDF file.", true);
    return;
  }

  showUploadStatus(`Reading & indexing "${file.name}"…`, false);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Upload failed (${res.status})`);
    }
    const data = await res.json();

    state.sessionId = data.session_id;
    state.filename = data.filename;

    el.docName.textContent = data.filename;
    el.docPages.textContent = data.num_pages;
    el.docChunks.textContent = data.num_chunks;

    el.docBadgeName.textContent = data.filename;
    el.docBadge.hidden = false;

    el.uploadPanel.hidden = true;
    el.docPanel.hidden = false;

    el.emptyState.hidden = true;
    switchTab("chat");
    resetWorkspace();
  } catch (err) {
    showUploadStatus(err.message, true);
  }
}

function showUploadStatus(message, isError) {
  el.uploadStatus.hidden = false;
  el.uploadStatus.classList.toggle("error", isError);
  el.uploadStatus.textContent = message;
}

el.removeDocBtn.addEventListener("click", async () => {
  if (state.sessionId) {
    fetch(`${API_BASE}/session/${state.sessionId}`, { method: "DELETE" }).catch(
      () => {}
    );
  }
  state.sessionId = null;
  state.filename = null;

  el.docPanel.hidden = true;
  el.uploadPanel.hidden = false;
  el.uploadStatus.hidden = true;
  el.fileInput.value = "";
  el.docBadge.hidden = true;

  el.emptyState.hidden = false;
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = true));

  resetWorkspace();
});

function resetWorkspace() {
  el.chatLog.querySelectorAll(".msg").forEach((m) => m.remove());
  el.blogOutput.innerHTML = `<p class="placeholder-text">Nothing generated yet — click "Generate blog post" above.</p>`;
  el.glossaryOutput.innerHTML = `<p class="placeholder-text">Nothing generated yet — click "Generate glossary" above.</p>`;
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

el.tabNav.addEventListener("click", (e) => {
  const btn = e.target.closest(".tab-btn");
  if (!btn) return;
  switchTab(btn.dataset.tab);
});

function switchTab(name) {
  document
    .querySelectorAll(".tab-btn")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document
    .querySelectorAll(".tab-panel")
    .forEach((p) => (p.hidden = p.id !== `tab-${name}`));
}

// ---------------------------------------------------------------------------
// Chat (RAG + memory)
// ---------------------------------------------------------------------------

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question || !requireSession()) return;

  appendUserMessage(question);
  el.chatInput.value = "";
  el.chatSendBtn.disabled = true;

  const sweepId = appendSweepIndicator("Retrieving relevant passages…");

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: question }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    removeSweepIndicator(sweepId);
    appendAssistantMessage(data.answer, data.sources);
  } catch (err) {
    removeSweepIndicator(sweepId);
    appendAssistantMessage(`Something went wrong: ${err.message}`, []);
  } finally {
    el.chatSendBtn.disabled = false;
  }
});

function appendUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-user";
  wrap.innerHTML = `<div class="msg-bubble"></div>`;
  wrap.querySelector(".msg-bubble").textContent = text;
  el.chatLog.appendChild(wrap);
  scrollChatToBottom();
}

function appendAssistantMessage(text, sources) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-assistant";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const toggle = document.createElement("div");
    toggle.className = "msg-sources";
    toggle.textContent = `▸ ${sources.length} source excerpt${
      sources.length > 1 ? "s" : ""
    }`;
    toggle.addEventListener("click", () => {
      toggle.classList.toggle("open");
      toggle.textContent = `${toggle.classList.contains("open") ? "▾" : "▸"} ${
        sources.length
      } source excerpt${sources.length > 1 ? "s" : ""}`;
    });

    const list = document.createElement("div");
    list.className = "msg-sources-list";
    sources.forEach((s) => {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      const pageLabel =
        s.page !== null && s.page !== undefined
          ? `Page ${s.page + 1}`
          : "Excerpt";
      chip.innerHTML = `<b>${pageLabel}</b> — ${escapeHtml(s.excerpt)}`;
      list.appendChild(chip);
    });

    wrap.appendChild(toggle);
    wrap.appendChild(list);
  }

  el.chatLog.appendChild(wrap);
  scrollChatToBottom();
}

let sweepCounter = 0;
function appendSweepIndicator(label) {
  const id = `sweep-${++sweepCounter}`;
  const wrap = document.createElement("div");
  wrap.className = "msg msg-assistant";
  wrap.id = id;
  wrap.innerHTML = `
       <div class="msg-bubble">
         <div class="sweep">
           <span>${escapeHtml(label)}</span>
           <span class="sweep-bar"></span>
         </div>
       </div>`;
  el.chatLog.appendChild(wrap);
  scrollChatToBottom();
  return id;
}

function removeSweepIndicator(id) {
  document.getElementById(id)?.remove();
}

function scrollChatToBottom() {
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

// ---------------------------------------------------------------------------
// Blog generation
// ---------------------------------------------------------------------------

el.generateBlogBtn.addEventListener("click", async () => {
  if (!requireSession()) return;

  el.generateBlogBtn.disabled = true;
  el.blogOutput.innerHTML = sweepHtml("Drafting the plain-English version…");

  try {
    const res = await fetch(`${API_BASE}/blog`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    el.blogOutput.innerHTML = markdownToHtml(data.blog_markdown);
  } catch (err) {
    el.blogOutput.innerHTML = `<p class="placeholder-text">Couldn't generate the blog post: ${escapeHtml(
      err.message
    )}</p>`;
  } finally {
    el.generateBlogBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Glossary generation
// ---------------------------------------------------------------------------

el.generateGlossaryBtn.addEventListener("click", async () => {
  if (!requireSession()) return;

  el.generateGlossaryBtn.disabled = true;
  el.glossaryOutput.innerHTML = sweepHtml("Pulling out key terms…");

  try {
    const res = await fetch(`${API_BASE}/glossary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderGlossary(data.glossary);
  } catch (err) {
    el.glossaryOutput.innerHTML = `<p class="placeholder-text">Couldn't generate the glossary: ${escapeHtml(
      err.message
    )}</p>`;
  } finally {
    el.generateGlossaryBtn.disabled = false;
  }
});

function renderGlossary(items) {
  if (!items || !items.length) {
    el.glossaryOutput.innerHTML = `<p class="placeholder-text">No terms were found.</p>`;
    return;
  }
  el.glossaryOutput.innerHTML = "";
  items.forEach(({ term, meaning }) => {
    const card = document.createElement("div");
    card.className = "glossary-card";
    card.innerHTML = `
         <p class="glossary-term">${escapeHtml(term)}</p>
         <p class="glossary-meaning">${escapeHtml(meaning)}</p>`;
    el.glossaryOutput.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function requireSession() {
  if (!state.sessionId) {
    alert("Upload a PDF first.");
    return false;
  }
  return true;
}

function sweepHtml(label) {
  return `<div class="sweep" style="padding:24px;">
       <span>${escapeHtml(label)}</span>
       <span class="sweep-bar"></span>
     </div>`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Minimal Markdown -> HTML converter (headings, bold, italic, paragraphs, lists). */
function markdownToHtml(md) {
  if (!md) return "";
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let inList = false;

  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };

  for (let raw of lines) {
    const line = raw.trim();

    if (!line) {
      closeList();
      continue;
    }

    let m;
    if ((m = line.match(/^#{1,2}\s+(.*)/))) {
      closeList();
      html += `<h1>${inlineMd(m[1])}</h1>`;
    } else if ((m = line.match(/^#{3,6}\s+(.*)/))) {
      closeList();
      html += `<h2>${inlineMd(m[1])}</h2>`;
    } else if ((m = line.match(/^[-*]\s+(.*)/))) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${inlineMd(m[1])}</li>`;
    } else {
      closeList();
      html += `<p>${inlineMd(line)}</p>`;
    }
  }
  closeList();
  return html;
}

function inlineMd(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}
