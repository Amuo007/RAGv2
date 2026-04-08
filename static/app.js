// ── State ──────────────────────────────────────────────────────────────────
let currentChatId = null;
let chats = [];
let streaming = false;
let serverSessionId = null;
let selectedArticle = null;

function showWelcome() {
  document.getElementById('welcome').style.display = '';
  document.getElementById('messagesArea').style.display = 'none';
}
function showChat() {
  document.getElementById('welcome').style.display = 'none';
  document.getElementById('messagesArea').style.display = '';
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  await Promise.all([loadSession(), loadModels(), loadChats()]);
}
init();

async function loadSession() {
  try {
    const data = await fetch('/session').then(r => r.json());
    serverSessionId = data.session_id;
  } catch(e) {
    console.error('Failed to load session', e);
  }
}

// ── Models ─────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await fetch('/models').then(r => r.json());
    const sel  = document.getElementById('modelSelect');
    sel.innerHTML = data.models.map(m =>
      `<option value="${esc(m)}">${esc(m)}</option>`
    ).join('');
  } catch(e) {
    console.error('Failed to load models', e);
  }
}

function getSelectedModel() {
  return document.getElementById('modelSelect').value;
}

function setSelectedModel(model) {
  const sel = document.getElementById('modelSelect');
  for (const opt of sel.options) {
    if (opt.value === model) { sel.value = model; return; }
  }
}

async function onModelChange() {
  if (!currentChatId || streaming) return;
  const model = getSelectedModel();
  try {
    const res = await fetch(`/chats/${currentChatId}`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model})
    }).then(r => r.json());

    const chat = chats.find(c => c.id === currentChatId);
    if (chat) {
      chat.model = model;
      chat.context_size = res.context_size;
      updateCtxBar(chat.tokens_used, res.context_size);
    }
  } catch(e) {
    console.error('Failed to update model', e);
  }
}

// ── Chat list ──────────────────────────────────────────────────────────────
async function loadChats() {
  try {
    const data = await fetch('/chats').then(r => r.json());
    chats = data.chats;
    renderChatList();
    if (chats.length > 0 && !currentChatId) {
      openChat(chats[0].id);
    }
  } catch(e) {
    console.error('Failed to load chats', e);
  }
}

function renderChatList() {
  const list = document.getElementById('chatList');
  if (chats.length === 0) {
    list.innerHTML = '<div class="no-chats">No chats yet</div>';
    return;
  }
  list.innerHTML = chats.map(c => `
    <div class="chat-item${c.id === currentChatId ? ' active' : ''}"
         onclick="openChat('${c.id}')">
      <span class="chat-item-title" title="${esc(c.title)}">${esc(c.title)}</span>
      <button class="chat-del-btn" onclick="deleteChat(event,'${c.id}')" title="Delete">✕</button>
    </div>
  `).join('');
}

// ── Open chat ──────────────────────────────────────────────────────────────
async function openChat(chatId) {
  if (streaming) return;
  currentChatId = chatId;
  renderChatList();

  const chat = chats.find(c => c.id === chatId);
  if (chat) {
    setSelectedModel(chat.model);
    updateCtxBar(chat.tokens_used, chat.context_size);
  }

  try {
    const data = await fetch(`/chats/${chatId}`).then(r => r.json());

    const idx = chats.findIndex(c => c.id === chatId);
    if (idx >= 0) chats[idx] = {...chats[idx], ...data.chat};

    showChat();

    const area = document.getElementById('messagesArea');
    area.innerHTML = '';
    for (const msg of data.messages) {
      if (msg.role === 'user') {
        appendUserMessage(msg.content, msg.is_rag);
      } else if (msg.role === 'assistant') {
        const sources = msg.sources ? JSON.parse(msg.sources) : [];
        const stats   = msg.stats   ? JSON.parse(msg.stats)   : null;
        appendAssistantMessage(msg.content, sources, stats);
      }
    }
    scrollBottom();

    const isStale = serverSessionId && data.chat.session_id && data.chat.session_id !== serverSessionId;
    setSessionStale(isStale);
    if (!isStale) document.getElementById('msgInput').focus();
  } catch(e) {
    console.error('Failed to open chat', e);
  }
}

// ── New chat ───────────────────────────────────────────────────────────────
async function newChat() {
  if (streaming) return;
  try {
    const model = getSelectedModel();
    const chat  = await fetch('/chats', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model})
    }).then(r => r.json());

    chats.unshift(chat);
    currentChatId = chat.id;
    renderChatList();
    document.getElementById('messagesArea').innerHTML = '';
    showChat();
    updateCtxBar(chat.tokens_used, chat.context_size);
    setSelectedModel(chat.model);
    setSessionStale(false);
    document.getElementById('msgInput').focus();
  } catch(e) {
    console.error('Failed to create chat', e);
  }
}

// ── Delete chat ────────────────────────────────────────────────────────────
async function deleteChat(e, chatId) {
  e.stopPropagation();
  if (streaming && chatId === currentChatId) return;
  if (!confirm('Delete this chat?')) return;

  try {
    await fetch(`/chats/${chatId}`, {method: 'DELETE'});
    chats = chats.filter(c => c.id !== chatId);

    if (currentChatId === chatId) {
      currentChatId = null;
      document.getElementById('messagesArea').innerHTML = '';
      showWelcome();
      updateCtxBar(0, null);
    }
    renderChatList();
  } catch(e) {
    console.error('Failed to delete chat', e);
  }
}

// ── Send message ───────────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('msgInput');
  const rawMessage = input.value.trim();
  if (!rawMessage || streaming) return;

  if (!currentChatId) {
    try {
      const model = getSelectedModel();
      const chat  = await fetch('/chats', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model})
      }).then(r => r.json());

      chats.unshift(chat);
      currentChatId = chat.id;
      document.getElementById('messagesArea').innerHTML = '';
      showChat();
      renderChatList();
      setSessionStale(false);
    } catch(e) {
      console.error('Failed to auto-create chat', e);
      return;
    }
  }

  // If article mode: build full message from chip + query
  let message = rawMessage;
  if (selectedArticle) {
    message = `/article ${selectedArticle} | ${rawMessage}`;
  }

  const isRag = message.toLowerCase().startsWith('/search');

  appendUserMessage(message, isRag ? 1 : 0);
  input.value = '';
  input.style.height = '';
  input.classList.remove('search-mode', 'article-mode');
  document.getElementById('searchBadge').classList.remove('on');

  // Clear article chip
  if (selectedArticle) {
    selectedArticle = null;
    document.getElementById('articleChipRow').style.display = 'none';
    input.placeholder = 'Message… (type /search or /article to query Wikipedia)';
  }
  document.getElementById('sendBtn').disabled = true;
  streaming = true;
  scrollBottom();

  const typingId = appendTyping();
  const bubbleEl = appendAssistantBubble();
  scrollBottom();

  const model = getSelectedModel();
  let buffer  = '';
  let full    = '';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({chat_id: currentChatId, message, model})
    });

    if (res.status === 409) {
      removeEl(typingId);
      bubbleEl.textContent = '[Session expired — start a new chat to continue]';
      setSessionStale(true);
      streaming = false;
      document.getElementById('sendBtn').disabled = true;
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();

    removeEl(typingId);

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const sentIdx = buffer.indexOf('\n\n[STATS]');
      if (sentIdx !== -1) {
        full = buffer.slice(0, sentIdx).replace(/\n+$/, '').trim();
        bubbleEl.textContent = full;
        try {
          const payload = JSON.parse(buffer.slice(sentIdx + '\n\n[STATS]'.length));
          finaliseAssistantMessage(bubbleEl, payload);
        } catch(err) { console.error('STATS parse error', err); }
        break;
      }
      full = buffer;
      bubbleEl.textContent = buffer;
      scrollBottom();
    }
  } catch(err) {
    removeEl(typingId);
    bubbleEl.textContent = `[Error: ${err.message}]`;
  }

  streaming = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('msgInput').focus();
  scrollBottom();
}

// ── DOM helpers ────────────────────────────────────────────────────────────
function appendUserMessage(content, isRag) {
  const area = document.getElementById('messagesArea');
  const row  = document.createElement('div');
  row.className = 'msg-row user';

  let inner = `<div class="msg-role">You</div>`;
  if (content.toLowerCase().startsWith('/article')) {
    inner += `<div class="search-tag article-tag">📄 ARTICLE</div>`;
  } else if (isRag) {
    inner += `<div class="search-tag">🔍 SEARCH</div>`;
  }
  inner += `<div class="msg-bubble">${escHtml(content)}</div>`;
  row.innerHTML = inner;
  area.appendChild(row);
}

function appendAssistantBubble() {
  const area = document.getElementById('messagesArea');
  const row  = document.createElement('div');
  row.className = 'msg-row assistant';
  row.innerHTML = `<div class="msg-role">Assistant</div><div class="msg-bubble"></div>`;
  area.appendChild(row);
  return row.querySelector('.msg-bubble');
}

function appendAssistantMessage(content, sources, stats) {
  const bubble = appendAssistantBubble();
  bubble.textContent = content;
  if (stats || (sources && sources.length)) {
    finaliseAssistantMessage(bubble, {...(stats||{}), sources: sources||[]});
  }
}

function finaliseAssistantMessage(bubbleEl, payload) {
  const row = bubbleEl.parentElement;

  // Stats row
  if (payload.prompt_tokens !== undefined) {
    const used = (payload.prompt_tokens || 0) + (payload.gen_tokens || 0);
    const statsDiv = document.createElement('div');
    statsDiv.className = 'msg-stats';

    let parts = [];
    if (payload.is_rag) {
      if (payload.embed_ms)    parts.push(`Embed <b>${payload.embed_ms}ms</b>`);
      if (payload.retrieve_ms) parts.push(`Retrieve <b>${payload.retrieve_ms}ms</b>`);
    }
    if (payload.ttft_ms)       parts.push(`TTFT <b>${payload.ttft_ms}ms</b>`);
    if (payload.gen_ms)        parts.push(`Gen <b>${payload.gen_ms}ms</b>`);
    if (payload.gen_tokens)    parts.push(`<b>${payload.gen_tokens}</b> tokens`);
    if (payload.total_ms)      parts.push(`Total <b>${payload.total_ms}ms</b>`);

    // ── Debug: classifier result ──────────────────────────────────────────
    if (payload.is_rag && payload.query_type) {
      const typeLabel = payload.query_type === 'person'
        ? `👤 person${payload.entity_name ? ` · <b>${escHtml(payload.entity_name)}</b>` : ''}`
        : `🔍 generic`;
      const fallbackLabel = payload.is_fallback
        ? ` · <span class="debug-fallback">fallback</span>`
        : '';
      parts.push(`<span class="debug-classifier">${typeLabel}${fallbackLabel}</span>`);
    }

    statsDiv.innerHTML = parts.map(p => `<span>${p}</span>`).join('');
    row.appendChild(statsDiv);

    // Update context bar
    if (currentChatId) {
      const chat = chats.find(c => c.id === currentChatId);
      if (chat) {
        chat.tokens_used = used;
        updateCtxBar(used, payload.context_size || chat.context_size);
      }
    }
  }

  // Update chat title in sidebar
  if (payload.new_title && payload.chat_id) {
    const chat = chats.find(c => c.id === payload.chat_id);
    if (chat) {
      chat.title = payload.new_title;
      renderChatList();
    }
  }

  // Fallback note
  if (payload.is_fallback && payload.is_rag !== false) {
    const note = document.createElement('div');
    note.className = 'fallback-note';
    note.textContent = 'ℹ️ No matching articles found in the ZIM file — this answer is from the model\'s own knowledge.';
    row.appendChild(note);
  }

  // Sources
  const sources = payload.sources || [];
  if (sources.length) {
    const uid = 'src-' + Math.random().toString(36).slice(2);
    const toggleDiv = document.createElement('div');
    toggleDiv.className = 'sources-toggle';
    toggleDiv.innerHTML = `▶ ${sources.length} source${sources.length>1?'s':''}`;
    toggleDiv.onclick = function() {
      const list = document.getElementById(uid);
      list.classList.toggle('open');
      this.innerHTML = (list.classList.contains('open') ? '▼ ' : '▶ ') +
        `${sources.length} source${sources.length>1?'s':''}`;
    };

    const listDiv = document.createElement('div');
    listDiv.className = 'sources-list';
    listDiv.id = uid;
    listDiv.innerHTML = sources.map(s => `
      <div class="source-card">
        <div class="source-card-head">
          <span class="source-title">${escHtml(s.title)}</span>
          <span class="source-score">score ${s.score}</span>
        </div>
        <div class="source-text">${escHtml(s.text)}</div>
      </div>
    `).join('');

    row.appendChild(toggleDiv);
    row.appendChild(listDiv);
  }
}

function appendTyping() {
  const area = document.getElementById('messagesArea');
  const row  = document.createElement('div');
  row.className = 'msg-row assistant';
  const id = 'typing-' + Date.now();
  row.id = id;
  row.innerHTML = `
    <div class="msg-role">Assistant</div>
    <div class="typing-bubble">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>`;
  area.appendChild(row);
  return id;
}

function removeEl(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollBottom() {
  const area = document.getElementById('messagesArea');
  area.scrollTop = area.scrollHeight;
}

// ── Context bar ────────────────────────────────────────────────────────────
function updateCtxBar(tokensUsed, contextSize) {
  const sec  = document.getElementById('ctxSection');
  const fill = document.getElementById('ctxFill');
  const text = document.getElementById('ctxText');

  if (!contextSize) {
    sec.classList.add('ctx-hidden');
    setContextFull(false);
    return;
  }
  sec.classList.remove('ctx-hidden');

  const pct  = Math.min(100, Math.round(tokensUsed / contextSize * 100));
  const left = Math.max(0, contextSize - tokensUsed);

  fill.style.width = pct + '%';
  fill.className = 'ctx-bar-fill' + (pct > 85 ? ' danger' : pct > 65 ? ' warn' : '');
  text.textContent = `${left.toLocaleString()} tokens left (${pct}% used)`;

  setContextFull(pct >= 100);
}

function setContextFull(isFull) {
  const input  = document.getElementById('msgInput');
  const btn    = document.getElementById('sendBtn');
  const banner = document.getElementById('ctxFullBanner');

  if (isFull) {
    input.disabled = true;
    input.placeholder = 'Context window full — start a new chat.';
    btn.disabled = true;
    banner.classList.add('on');
  } else {
    input.disabled = false;
    input.placeholder = 'Message… (type /search or /article to query Wikipedia)';
    if (!streaming) btn.disabled = false;
    banner.classList.remove('on');
  }
}

function setSessionStale(isStale) {
  const input  = document.getElementById('msgInput');
  const btn    = document.getElementById('sendBtn');
  const banner = document.getElementById('ctxFullBanner');

  if (isStale) {
    input.disabled = true;
    input.placeholder = 'Session ended — start a new chat to continue.';
    btn.disabled = true;
    banner.classList.add('on');
    banner.textContent = 'Server was restarted — this session is no longer available. Please start a new chat.';
  } else {
    input.disabled = false;
    input.placeholder = 'Message… (type /search or /article to query Wikipedia)';
    if (!streaming) btn.disabled = false;
    banner.classList.remove('on');
    banner.textContent = 'Context window full — please start a new chat to continue.';
  }
}

// ── Slash-command autocomplete ─────────────────────────────────────────────
const COMMANDS = [
  { name: '/search',  desc: 'Semantic search across all Wikipedia articles' },
  { name: '/article', desc: 'Pin a specific article, then ask about it' },
];

let cmdSelectedIdx = -1;

function onInput(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';

  const val = el.value;

  // Article selected — input is purely the query, no mode switching
  if (selectedArticle) {
    closeCmdPopup();
    closeTitlePopup();
    return;
  }

  const isSearch  = val.toLowerCase().startsWith('/search');
  const isArticle = val.toLowerCase().startsWith('/article');
  el.classList.toggle('search-mode',  isSearch && !isArticle);
  el.classList.toggle('article-mode', isArticle);
  document.getElementById('searchBadge').classList.toggle('on', isSearch && !isArticle);

  // Slash-command popup: only when the value is a bare /word with no space yet
  if (/^(\/\S*)$/.test(val)) {
    const typed = val.toLowerCase();
    renderCmdPopup(COMMANDS.filter(c => c.name.startsWith(typed)));
    closeTitlePopup();
    return;
  }

  closeCmdPopup();

  // Title autocomplete while user is typing the article name
  if (isArticle) {
    const term = val.replace(/^\/article\s*/i, '').trim();
    if (term.length >= 1) {
      debouncedFetchTitles(term);
    } else {
      closeTitlePopup();
    }
  } else {
    closeTitlePopup();
  }
}

function renderCmdPopup(matches) {
  const popup = document.getElementById('cmdPopup');
  cmdSelectedIdx = -1;
  if (matches.length === 0) { closeCmdPopup(); return; }
  popup.innerHTML = matches.map((c, i) => `
    <div class="cmd-item" data-cmd="${esc(c.name)}"
         onmousedown="pickCmd('${esc(c.name)}')"
         onmouseover="hoverCmd(${i})">
      <span class="cmd-name">${esc(c.name)}</span>
      <span class="cmd-desc">${esc(c.desc)}</span>
    </div>
  `).join('');
  popup.classList.add('open');
}

function closeCmdPopup() {
  const popup = document.getElementById('cmdPopup');
  popup.classList.remove('open');
  popup.innerHTML = '';
  cmdSelectedIdx = -1;
}

function hoverCmd(idx) {
  cmdSelectedIdx = idx;
  document.querySelectorAll('.cmd-item').forEach((el, i) =>
    el.classList.toggle('selected', i === idx)
  );
}

function pickCmd(name) {
  const input = document.getElementById('msgInput');
  input.value = name + ' ';
  input.focus();
  onInput(input);
  closeCmdPopup();
}

function handleKey(e) {
  // Title popup takes priority
  const titlePopup = document.getElementById('titlePopup');
  if (titlePopup && titlePopup.classList.contains('open')) {
    const items = titlePopup.querySelectorAll('.title-item');
    if (items.length) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        titleSelectedIdx = Math.min(titleSelectedIdx + 1, items.length - 1);
        items.forEach((el, i) => el.classList.toggle('selected', i === titleSelectedIdx));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        titleSelectedIdx = Math.max(titleSelectedIdx - 1, 0);
        items.forEach((el, i) => el.classList.toggle('selected', i === titleSelectedIdx));
        return;
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && titleSelectedIdx >= 0)) {
        e.preventDefault();
        const chosen = items[titleSelectedIdx >= 0 ? titleSelectedIdx : 0];
        if (chosen) pickTitle(chosen.dataset.title);
        return;
      }
      if (e.key === 'Escape') { closeTitlePopup(); return; }
    }
  }

  const popup = document.getElementById('cmdPopup');
  const isOpen = popup.classList.contains('open');
  const items  = popup.querySelectorAll('.cmd-item');

  if (isOpen && items.length) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      cmdSelectedIdx = Math.min(cmdSelectedIdx + 1, items.length - 1);
      items.forEach((el, i) => el.classList.toggle('selected', i === cmdSelectedIdx));
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      cmdSelectedIdx = Math.max(cmdSelectedIdx - 1, 0);
      items.forEach((el, i) => el.classList.toggle('selected', i === cmdSelectedIdx));
      return;
    }
    if (e.key === 'Tab' || (e.key === 'Enter' && cmdSelectedIdx >= 0)) {
      e.preventDefault();
      const chosen = items[cmdSelectedIdx >= 0 ? cmdSelectedIdx : 0];
      if (chosen) pickCmd(chosen.dataset.cmd);
      return;
    }
    if (e.key === 'Escape') { closeCmdPopup(); return; }
  }

  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// ── Utils ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function esc(str) { return escHtml(str); }

// ── Article title autocomplete ─────────────────────────────────────────────
let titleSelectedIdx = -1;
let _titleDebounce   = null;

function debouncedFetchTitles(q) {
  clearTimeout(_titleDebounce);
  _titleDebounce = setTimeout(() => fetchTitles(q), 160);
}

async function fetchTitles(q) {
  try {
    const data = await fetch(`/titles?q=${encodeURIComponent(q)}`).then(r => r.json());
    renderTitlePopup(data.titles || []);
  } catch(e) {
    closeTitlePopup();
  }
}

function renderTitlePopup(titles) {
  const popup = document.getElementById('titlePopup');
  titleSelectedIdx = -1;
  if (!titles.length) { closeTitlePopup(); return; }
  popup.innerHTML = titles.map((t, i) => `
    <div class="cmd-item title-item" data-title="${esc(t)}"
         onmousedown="pickTitle('${esc(t).replace(/'/g, '&#39;')}')"
         onmouseover="hoverTitle(${i})">
      <span class="cmd-name title-name">📄 ${esc(t)}</span>
    </div>
  `).join('');
  popup.classList.add('open');
}

function closeTitlePopup() {
  const popup = document.getElementById('titlePopup');
  if (!popup) return;
  popup.classList.remove('open');
  popup.innerHTML = '';
  titleSelectedIdx = -1;
}

function hoverTitle(idx) {
  titleSelectedIdx = idx;
  document.querySelectorAll('.title-item').forEach((el, i) =>
    el.classList.toggle('selected', i === idx)
  );
}

function pickTitle(title) {
  selectedArticle = title;
  const input = document.getElementById('msgInput');
  input.value = '';
  input.placeholder = `Ask about "${title}"…`;
  input.classList.remove('article-mode');
  input.classList.add('article-mode');
  input.focus();
  closeTitlePopup();
  closeCmdPopup();
  document.getElementById('articleChipTitle').textContent = title;
  document.getElementById('articleChipRow').style.display = '';
}

function clearArticle() {
  selectedArticle = null;
  const input = document.getElementById('msgInput');
  input.value = '';
  input.placeholder = 'Message… (type /search or /article to query Wikipedia)';
  input.classList.remove('article-mode');
  document.getElementById('articleChipRow').style.display = 'none';
  input.focus();
}