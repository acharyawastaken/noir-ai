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
  var [messages, setMessages] = useState([]);
  var [input, setInput] = useState("");
  var [busy, setBusy] = useState(false);
  var [file, setFile] = useState(null);
  var [status, setStatus] = useState("idle");
  var [statusText, setStatusText] = useState("Waiting for document\u2026");
  var [logText, setLogText] = useState("");
  var [showLog, setShowLog] = useState(false);
  var [showFeatures, setShowFeatures] = useState(false);
  var [latestAssistantId, setLatestAssistantId] = useState(null);

  var fileInputRef = useRef(null);
  var textareaRef = useRef(null);
  var messagesEndRef = useRef(null);

  // Clear backend index on refresh / mount
  useEffect(function () {
    fetch("/reset", { method: "POST" })
      .then(function () {
        console.log("Session reset complete.");
      })
      .catch(function (err) {
        console.error("Session reset failed:", err);
      });
  }, []);

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
      setStatus("error");
      setStatusText("Unsupported format: " + ext);
      return;
    }
    setFile(f);
    setStatus("loading");
    setStatusText("Ingesting \"" + f.name + "\"\u2026");
    setLogText("Starting ingestion pipeline for " + f.name + "\u2026");

    var formData = new FormData();
    formData.append("file", f);

    fetch("/upload", { method: "POST", body: formData })
      .then(function (res) { return res.json().then(function (data) { return { ok: res.ok, data: data }; }); })
      .then(function (result) {
        if (result.ok) {
          setStatus("ready");
          setStatusText('"' + f.name + '" indexed \u2713');
          setLogText(result.data.details || "Done.");
        } else {
          setStatus("error");
          setStatusText("Ingestion failed");
          setLogText("ERROR: " + (result.data.detail || ""));
        }
      })
      .catch(function (err) {
        setStatus("error");
        setStatusText("API unreachable");
        setLogText("ERROR: " + err.message);
      });
  }

  // ---- Send query ----
  var sendQuery = useCallback(function () {
    var q = input.trim();
    if (!q || busy) return;

    setInput("");
    setBusy(true);

    var userMsg = { id: Date.now().toString(), role: "user", text: q };
    setMessages(function (prev) { return [].concat(prev, [userMsg]); });
    scrollToBottom();

    fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    })
      .then(function (res) { return res.json().then(function (data) { return { ok: res.ok, data: data }; }); })
      .then(function (result) {
        var answer = result.data.response || "";
        var botId = (Date.now() + 1).toString();
        var botMsg = {
          id: botId,
          role: "assistant",
          text: result.ok ? answer : "\u26A0 " + (result.data.detail || "Something went wrong."),
        };
        setLatestAssistantId(botId);
        setMessages(function (prev) { return [].concat(prev, [botMsg]); });
        setBusy(false);
        scrollToBottom();
      })
      .catch(function () {
        var errId = (Date.now() + 1).toString();
        setMessages(function (prev) {
          return [].concat(prev, [{
            id: errId,
            role: "assistant",
            text: "\u26A0 Cannot reach the API. Is the FastAPI server running?",
          }]);
        });
        setLatestAssistantId(errId);
        setBusy(false);
        scrollToBottom();
      });
  }, [input, busy, scrollToBottom]);

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
  return (
    <div className="app-shell">
      <BGPattern />

      {/* HEADER */}
      <header className="header">
        <div className="header-brand">
          <img src="/logo.png" className="header-logo" alt="logo" />
          <div>
            <div className="header-title">noir</div>
            <div className="header-subtitle">Multi-Source Hybrid RAG</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <button
            className="btn btn-ghost"
            onClick={function () { setShowFeatures(true); }}
            style={{ fontSize: "9px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}
          >
            <Icon name="check-square" size={13} />
            Features
          </button>
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
            <span style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {statusText}
            </span>
          </div>
        </div>
      </header>

      {/* FEATURES MODAL */}
      {showFeatures && (
        <div className="modal-overlay" onClick={function () { setShowFeatures(false); }}>
          <div className="modal-content" onClick={function (e) { e.stopPropagation(); }}>
            <div className="modal-header">
              <div className="modal-title">
                <Icon name="list-checks" size={18} />
                Features Checklist
              </div>
              <button className="modal-close" onClick={function () { setShowFeatures(false); }}>
                <Icon name="x" size={16} />
              </button>
            </div>
            
            <div className="feature-group">
              <div className="feature-group-title">Achieved (MVP)</div>
              <div className="feature-list">
                {[
                  "CLI Scripts (ingest & query)",
                  "File Support: Markdown, PDF, CSV, Excel, Images (OCR)",
                  "Hybrid Retrieval (Vector + BM25)",
                  "Document Summarization",
                  "Witty/Dry-witted AI Personality (Noir)",
                  "Fallback general conversation"
                ].map(function (f, i) {
                  return (
                    <div className="feature-item" key={"ach-" + i}>
                      <div className="feature-checkbox checked"><Icon name="check" size={12} /></div>
                      <div className="feature-text">{f}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="feature-group" style={{ marginBottom: 0 }}>
              <div className="feature-group-title">Next Phase (Pending)</div>
              <div className="feature-list">
                {[
                  "Multi-Agent RAG Orchestration",
                  "Authentication with JWT tokens",
                  "Citations, Reranking & Query Expansion",
                  "Multi-Doc index query routing & PPTX parsing",
                  "Chat history and memory stores"
                ].map(function (f, i) {
                  return (
                    <div className="feature-item" key={"pend-" + i}>
                      <div className="feature-checkbox"></div>
                      <div className="feature-text pending">{f}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

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
                    <button className="btn btn-ghost" onClick={function () { setMessages([]); setLatestAssistantId(null); }}>
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
  );
}

// ── Mount ──
var root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
