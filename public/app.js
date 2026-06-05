/* ============================================================
   RUIXEN RAG — React App
   Typewriter effect · Grid BG · Cascadia Code
   ============================================================ */

const { useState, useRef, useCallback, useEffect } = React;

const ALLOWED_EXTS = [".pdf", ".docx", ".doc", ".csv", ".xlsx", ".md", ".txt", ".png", ".jpg", ".jpeg"];



// ── Icon helper ──
function Icon({ name, size = 16 }) {
  const ref = useRef(null);
  useEffect(function () {
    if (ref.current && window.lucide) {
      ref.current.innerHTML = "";
      var i = document.createElement("i");
      i.setAttribute("data-lucide", name);
      i.style.width = size + "px";
      i.style.height = size + "px";
      ref.current.appendChild(i);
      window.lucide.createIcons({ nodes: [i] });
    }
  }, [name, size]);
  return <span ref={ref} style={{ display: "inline-flex", alignItems: "center", lineHeight: 0 }} />;
}

// ── Typewriter Text Component ──
// Renders text character-by-character with staggered reveal animation
function TypewriterText({ text, speed = 35 }) { // Reduced speed (increased delay to 35ms) for lively animation
  var ref = useRef(null);
  var [rendered, setRendered] = useState(false);

  useEffect(function () {
    if (rendered) return;
    setRendered(true);
  }, []);

  if (!text) return null;

  // Split text into words, preserving whitespace
  var words = text.split(/(\s+)/);
  var charIndex = 0;

  return (
    <span className="typewriter-text" ref={ref}>
      {words.map(function (word, wIdx) {
        if (/^\s+$/.test(word)) {
          // Whitespace — render as-is but still increment index
          var wsChars = word.split("").map(function (ch, i) {
            charIndex++;
            return <span key={"ws-" + wIdx + "-" + i}>{ch}</span>;
          });
          return <span key={"space-" + wIdx}>{wsChars}</span>;
        }
        var chars = word.split("").map(function (char, cIdx) {
          var delay = charIndex * speed;
          charIndex++;
          return (
            <span
              className="typewriter-char"
              key={"c-" + wIdx + "-" + cIdx}
              style={{ animationDelay: delay + "ms" }}
            >
              {char}
            </span>
          );
        });
        return <span className="typewriter-word" key={"w-" + wIdx}>{chars}</span>;
      })}
      {rendered && <span className="typewriter-cursor" />}
    </span>
  );
}

// ── Typing indicator ──
function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span /><span /><span />
    </div>
  );
}

// ── Message Bubble with typewriter for assistant ──
function MessageBubble({ msg, isLatest }) {
  return (
    <div className={"msg " + msg.role + " msg-enter"}>
      {msg.role === "assistant" && (
        <div className="msg-avatar">{"\u2726"}</div>
      )}
      <div className="msg-bubble">
        <div style={{ whiteSpace: "pre-wrap" }}>
          {msg.role === "assistant" && isLatest ? (
            <TypewriterText text={msg.text} speed={32} />
          ) : (
            msg.text
          )}
        </div>
      </div>
      {msg.role === "user" && (
        <div className="msg-avatar" style={{ background: "rgba(255,255,255,0.04)" }}>U</div>
      )}
    </div>
  );
}

// ── Quick Action Pill ──
function QuickAction({ icon, label, onClick }) {
  return (
    <button className="quick-action-btn" onClick={onClick}>
      <Icon name={icon} size={13} />
      {label}
    </button>
  );
}

// ── Background Pattern ──
function BGPattern() {
  return <div className="bg-pattern" />;
}

// ============================================================
// MAIN APP
// ============================================================
function App() {
  var [chats, setChats] = useState(function () {
    var stored = localStorage.getItem("noir_chats");
    if (stored) {
      try {
        var parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      } catch (e) {}
    }
    return [{ id: "chat-1", name: "New Chat 1" }];
  });

  var [activeChatId, setActiveChatId] = useState(function () {
    var stored = localStorage.getItem("noir_active_chat_id");
    return stored || "chat-1";
  });

  var [chatStores, setChatStores] = useState(function () {
    var stored = localStorage.getItem("noir_chat_stores");
    if (stored) {
      try {
        var parsed = JSON.parse(stored);
        if (parsed && typeof parsed === "object") return parsed;
      } catch (e) {}
    }
    return {
      "chat-1": { messages: [], file: null, status: "idle", statusText: "Waiting for document\u2026", logText: "" }
    };
  });

  var [input, setInput] = useState("");
  var [busy, setBusy] = useState(false);
  var [showLog, setShowLog] = useState(false);
  var [latestAssistantId, setLatestAssistantId] = useState(null);

  // Authentication State
  var [token, setToken] = useState(localStorage.getItem("noir_token") || "");
  var [username, setUsername] = useState("");
  var [password, setPassword] = useState("");
  var [loginError, setLoginError] = useState("");

  var fileInputRef = useRef(null);
  var textareaRef = useRef(null);
  var messagesEndRef = useRef(null);

  // Persist to localStorage
  useEffect(function () {
    localStorage.setItem("noir_chats", JSON.stringify(chats));
  }, [chats]);

  useEffect(function () {
    localStorage.setItem("noir_active_chat_id", activeChatId);
  }, [activeChatId]);

  useEffect(function () {
    localStorage.setItem("noir_chat_stores", JSON.stringify(chatStores));
  }, [chatStores]);

  // Derived state for the active chat
  var currentStore = chatStores[activeChatId] || { messages: [], file: null, status: "idle", statusText: "Waiting for document\u2026", logText: "" };
  var messages = currentStore.messages || [];
  var file = currentStore.file || null;
  var status = currentStore.status || "idle";
  var statusText = currentStore.statusText || "Waiting for document\u2026";
  var logText = currentStore.logText || "";

  function updateActiveStore(updates) {
    setChatStores(function (prev) {
      var current = prev[activeChatId] || { messages: [], file: null, status: "idle", statusText: "Waiting...", logText: "" };
      var updated = Object.assign({}, current, updates);
      var next = Object.assign({}, prev);
      next[activeChatId] = updated;
      return next;
    });
  }

  function appendMessageToActiveStore(msg) {
    setChatStores(function (prev) {
      var current = prev[activeChatId] || { messages: [], file: null, status: "idle", statusText: "Waiting...", logText: "" };
      var updatedMsgs = [].concat(current.messages || [], [msg]);
      var updated = Object.assign({}, current, { messages: updatedMsgs });
      var next = Object.assign({}, prev);
      next[activeChatId] = updated;
      return next;
    });
  }

  function handleNewChat() {
    if (chats.length >= 10) {
      alert("Limit of 10 chats reached. Please delete an existing chat first.");
      return;
    }
    var nextId = "chat-" + Date.now();
    var nextName = "New Chat " + (chats.length + 1);
    
    setChats(function (prev) {
      return prev.concat([{ id: nextId, name: nextName }]);
    });
    
    setChatStores(function (prev) {
      var next = Object.assign({}, prev);
      next[nextId] = { messages: [], file: null, status: "idle", statusText: "Waiting for document\u2026", logText: "" };
      return next;
    });
    
    setActiveChatId(nextId);
  }

  function handleDeleteChat(chatId, e) {
    if (e) e.stopPropagation();
    if (chats.length <= 1) {
      alert("You must keep at least one active chat.");
      return;
    }
    var nextChats = chats.filter(function (c) { return c.id !== chatId; });
    setChats(nextChats);
    
    setChatStores(function (prev) {
      var next = Object.assign({}, prev);
      delete next[chatId];
      return next;
    });
    
    if (activeChatId === chatId) {
      setActiveChatId(nextChats[0].id);
    }
  }

  // Clear backend index on refresh / mount (only if authenticated)
  useEffect(function () {
    if (!token) return;
    fetch("/reset", {
      method: "POST",
      headers: { "Authorization": "Bearer " + token }
    })
      .then(function () {
        console.log("Session reset complete.");
      })
      .catch(function (err) {
        console.error("Session reset failed:", err);
      });
  }, [token]);

  function handleLogin(e) {
    if (e) e.preventDefault();
    setLoginError("");
    fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: username, password: password })
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (data) {
            throw new Error(data.detail || "Authentication failed");
          });
        }
        return res.json();
      })
      .then(function (data) {
        localStorage.setItem("noir_token", data.token);
        localStorage.setItem("noir_user", data.username);
        setToken(data.token);
        setUsername("");
        setPassword("");
      })
      .catch(function (err) {
        setLoginError(err.message);
      });
  }

  function handleLogout() {
    localStorage.removeItem("noir_token");
    localStorage.removeItem("noir_user");
    localStorage.removeItem("noir_chats");
    localStorage.removeItem("noir_active_chat_id");
    localStorage.removeItem("noir_chat_stores");
    setToken("");
    setChats([{ id: "chat-1", name: "New Chat 1" }]);
    setActiveChatId("chat-1");
    setChatStores({
      "chat-1": { messages: [], file: null, status: "idle", statusText: "Waiting for document\u2026", logText: "" }
    });
  }

  var scrollToBottom = useCallback(function () {
    setTimeout(function () {
      if (messagesEndRef.current) messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }, 80);
  }, []);

  var autoResize = useCallback(function () {
    var el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  }, []);

  // ---- File handling (Auto-Ingestion) ----
  function handleFile(f) {
    if (!f) return;
    var ext = "." + f.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      updateActiveStore({
        status: "error",
        statusText: "Unsupported format: " + ext
      });
      return;
    }
    updateActiveStore({
      file: { name: f.name },
      status: "loading",
      statusText: "Ingesting \"" + f.name + "\"\u2026",
      logText: "Starting ingestion pipeline for " + f.name + "\u2026"
    });

    var formData = new FormData();
    formData.append("file", f);

    fetch("/upload", {
      method: "POST",
      body: formData,
      headers: { "Authorization": "Bearer " + token }
    })
      .then(function (res) { return res.json().then(function (data) { return { ok: res.ok, data: data }; }); })
      .then(function (result) {
        if (result.ok) {
          updateActiveStore({
            status: "ready",
            statusText: '"' + f.name + '" indexed \u2713',
            logText: result.data.details || "Done."
          });
        } else {
          updateActiveStore({
            status: "error",
            statusText: "Ingestion failed",
            logText: "ERROR: " + (result.data.detail || "")
          });
        }
      })
      .catch(function (err) {
        updateActiveStore({
          status: "error",
          statusText: "API unreachable",
          logText: "ERROR: " + err.message
        });
      });
  }

  // ---- Send query ----
  var sendQuery = useCallback(function () {
    var q = input.trim();
    if (!q || busy) return;

    setInput("");
    setBusy(true);

    var userMsg = { id: Date.now().toString(), role: "user", text: q };
    appendMessageToActiveStore(userMsg);
    scrollToBottom();

    fetch("/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      },
      body: JSON.stringify({ query: q, session_id: activeChatId }),
    })
      .then(function (res) {
        if (res.status === 401) {
          handleLogout();
          throw new Error("Session expired. Please log in again.");
        }
        return res.json().then(function (data) { return { ok: res.ok, data: data }; });
      })
      .then(function (result) {
        var answer = result.data.response || "";
        var botId = (Date.now() + 1).toString();
        var botMsg = {
          id: botId,
          role: "assistant",
          text: result.ok ? answer : "\u26A0 " + (result.data.detail || "Something went wrong."),
        };
        setLatestAssistantId(botId);
        appendMessageToActiveStore(botMsg);
        
        // Auto-rename chat using dynamic generated title from description and reply
        if (result.ok && result.data.title) {
          var newTitle = result.data.title;
          setChats(function (prev) {
            return prev.map(function (c) {
              if (c.id === activeChatId) {
                return Object.assign({}, c, { name: newTitle });
              }
              return c;
            });
          });
        }

        setBusy(false);
        scrollToBottom();
      })
      .catch(function () {
        var errId = (Date.now() + 1).toString();
        appendMessageToActiveStore({
          id: errId,
          role: "assistant",
          text: "\u26A0 Cannot reach the API. Is the FastAPI server running?",
        });
        setLatestAssistantId(errId);
        setBusy(false);
        scrollToBottom();
      });
  }, [input, busy, scrollToBottom, token, activeChatId, messages.length]);

  function handleQuickAction(text) {
    setInput(text);
    setTimeout(function () {
      autoResize();
      if (textareaRef.current) textareaRef.current.focus();
    }, 10);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  }

  // ============================================================
  // RENDER
  // ============================================================
  if (!token) {
    return (
      <div className="auth-overlay">
        <BGPattern />
        <div className="auth-card">
          <img src="/logo.png" className="auth-logo" alt="logo" />
          <h1 className="auth-title">noir</h1>
          <p className="auth-subtitle">RAG Engine — Authentication Required</p>
          <form onSubmit={handleLogin}>
            <div className="auth-form-group">
              <label className="auth-label">Username</label>
              <input
                type="text"
                className="auth-input"
                value={username}
                onChange={function (e) { setUsername(e.target.value); }}
                placeholder="Enter username (e.g. admin)"
                required
              />
            </div>
            <div className="auth-form-group">
              <label className="auth-label">Password</label>
              <input
                type="password"
                className="auth-input"
                value={password}
                onChange={function (e) { setPassword(e.target.value); }}
                placeholder="Enter password"
                required
              />
            </div>
            <button type="submit" className="auth-btn">Log In</button>
          </form>
          {loginError && <div className="auth-error">{loginError}</div>}
        </div>
      </div>
    );
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <img src="/logo.png" className="sidebar-logo" alt="logo" />
          <span className="sidebar-title">noir</span>
        </div>

        <button className="sidebar-new-chat-btn" onClick={handleNewChat}>
          <Icon name="plus" size={14} style={{ marginRight: "6px" }} />
          New Chat
        </button>

        <div className="sidebar-chats-container">
          <div className="sidebar-section-title">Recent Chats ({chats.length}/10)</div>
          <div className="sidebar-chats-list">
            {chats.map(function (c) {
              var isActive = c.id === activeChatId;
              return (
                <div
                  key={c.id}
                  className={"sidebar-chat-item " + (isActive ? "active" : "")}
                  onClick={function () { setActiveChatId(c.id); }}
                >
                  <div style={{ display: "flex", alignItems: "center", overflow: "hidden", flex: 1 }}>
                    <Icon name="message-square" size={12} style={{ marginRight: "8px", opacity: 0.6, flexShrink: 0 }} />
                    <span className="chat-name">{c.name}</span>
                  </div>
                  <button className="chat-delete-btn" onClick={function (e) { handleDeleteChat(c.id, e); }}>
                    <Icon name="trash-2" size={11} />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="avatar">{(localStorage.getItem("noir_user") || "U").substring(0, 1).toUpperCase()}</div>
            <span className="username">{localStorage.getItem("noir_user") || "User"}</span>
          </div>
          <button className="sidebar-logout-btn" onClick={handleLogout} title="Log Out">
            <Icon name="log-out" size={13} />
          </button>
        </div>
      </aside>

      <div className="app-shell">
        <BGPattern />

        {/* HEADER */}
        <header className="header">
          <div className="header-brand">
            <div>
              <div className="header-title" style={{ fontSize: "14px", fontWeight: "600" }}>
                {chats.find(function (c) { return c.id === activeChatId; })?.name || "Chat"}
              </div>
              <div className="header-subtitle" style={{ fontSize: "9px" }}>Multi-Source Hybrid RAG</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {logText && (
              <button
                className="btn btn-ghost"
                onClick={function () { setShowLog(function (s) { return !s; }); }}
                style={{ fontSize: "9px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}
              >
                <Icon name="terminal" size={13} />
                {showLog ? "Hide" : "Logs"}
              </button>
            )}
            <div className={"header-status " + status}>
              <span className="dot" />
              <span style={{ maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {statusText}
              </span>
            </div>
          </div>
        </header>

        {/* LOG PANEL */}
        {showLog && logText && (
          <div style={{ width: "100%", position: "relative", zIndex: 2 }}>
            <div className="chat-container">
              <div className="log-panel fade-in">
                <div className="log-panel-header">
                  <Icon name="terminal" size={11} />
                  Pipeline Logs
                </div>
                {logText}
              </div>
            </div>
          </div>
        )}

        {/* CHAT AREA */}
        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="chat-container landing fade-in">
              <img src="/logo.png" style={{ width: "60px", height: "60px", borderRadius: "18px", border: "1px solid var(--border-subtle)", objectFit: "cover" }} alt="logo" />
              <h1>noir</h1>
              <p>
                Upload a document, ingest it into the hybrid index, then ask anything. Answers are synthesized from both semantic and keyword search.
              </p>
              <div className="landing-formats">
                {["PDF", "DOCX", "DOC", "CSV", "XLSX", "MD", "TXT", "PNG", "JPG", "JPEG"].map(function (f) {
                  return <span className="format-badge" key={f}>{f}</span>;
                })}
              </div>
            </div>
          ) : (
            <div className="messages">
              <div className="chat-container">
                {messages.map(function (m) {
                  return <MessageBubble key={m.id} msg={m} isLatest={m.id === latestAssistantId} />;
                })}
                {busy && (
                  <div className="msg assistant msg-enter">
                    <div className="msg-avatar">{"\u2726"}</div>
                    <div className="msg-bubble" style={{ borderStyle: "dashed", borderColor: "rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.01)" }}>
                      <TypingIndicator />
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          )}

          {/* INPUT AREA */}
          <div className="input-area">
            <div className="chat-container">
              {messages.length === 0 && (
                <div className="quick-actions fade-in">
                  <QuickAction icon="upload" label="Upload File" onClick={function () { if (fileInputRef.current) fileInputRef.current.click(); }} />
                  <QuickAction icon="file-text" label="Summarize" onClick={function () { handleQuickAction("Summarize this document"); }} />
                  <QuickAction icon="list" label="Key Topics" onClick={function () { handleQuickAction("What are the key topics?"); }} />
                  <QuickAction icon="calendar" label="Dates" onClick={function () { handleQuickAction("What dates are mentioned?"); }} />
                  <QuickAction icon="users" label="Names" onClick={function () { handleQuickAction("List all mentioned names"); }} />
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.doc,.csv,.xlsx,.md,.txt,.png,.jpg,.jpeg"
                style={{ display: "none" }}
                onChange={function (e) { handleFile(e.target.files ? e.target.files[0] : null); }}
              />

              <div className="input-box">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={function (e) { setInput(e.target.value); autoResize(); }}
                  onKeyDown={handleKeyDown}
                  placeholder={status === "ready" ? "Ask anything about your document\u2026" : "Upload a document to begin\u2026"}
                  rows={1}
                  disabled={busy}
                />
                <div className="input-controls">
                  <div className="input-controls-left">
                    <button
                      className="btn btn-icon"
                      onClick={function () { if (fileInputRef.current) fileInputRef.current.click(); }}
                      disabled={status === "loading" || busy}
                      title="Attach file"
                    >
                      <Icon name="paperclip" size={15} />
                    </button>
                    {file && (
                      <div className="file-chip">
                        <span className="name">{"\uD83D\uDCCE " + file.name}</span>
                        {status === "loading" && <span className="status-indicator loading">...</span>}
                        {status === "ready" && <span className="status-indicator success">✓</span>}
                        {status === "error" && <span className="status-indicator error">✗</span>}
                      </div>
                    )}
                  </div>
                  <div className="input-controls-right">
                    {messages.length > 0 && (
                      <button className="btn btn-ghost" onClick={function () { updateActiveStore({ messages: [] }); setLatestAssistantId(null); }}>
                        Clear
                      </button>
                    )}
                    <button
                      className="btn btn-send"
                      onClick={sendQuery}
                      disabled={busy || !input.trim()}
                    >
                      <Icon name="arrow-up" size={15} />
                      Send
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Mount ──
var root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
