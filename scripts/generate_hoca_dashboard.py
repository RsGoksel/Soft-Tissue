"""
Generate a single-file dashboard + chatbot HTML for the advisor.

The output `hoca_dashboard.html` contains ALL relevant project markdown
embedded as JavaScript context, a chat UI calling the xAI Grok API
directly, and a localStorage-based 10-questions-per-day rate limiter.

The generated HTML embeds an API key. **DO NOT COMMIT** the output file —
it is in `.gitignore`. The advisor receives the HTML via
direct share (WhatsApp / Google Drive) and opens it locally in any
modern browser.

Usage:
    # put the key in .grok_key (single line, gitignored) ---
    python scripts/generate_hoca_dashboard.py

    # or pass via env -----------------------------------------
    XAI_API_KEY=xai-... python scripts/generate_hoca_dashboard.py

Output:
    hoca_dashboard.html        (project root, gitignored)
"""
from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "hoca_dashboard.html"
KEY_PATH = ROOT / ".grok_key"

# Markdown files to embed as project knowledge base.
# Each entry: (display label shown to the model, file path).
CONTEXT_FILES = [
    ("README (proje özeti)",                  "README.md"),
    ("Teknik Detaylar (tam belge)",           "reports/technical_details.md"),
    ("Fizik Brifingi",                        "reports/physics_first_brief.md"),
    ("Model Girdileri ve Normalizasyon",      "reports/inputs_and_normalization.md"),
    ("Literatür Notları (IEEE)",              "reports/literature_notes.md"),
    ("Pazartesi Toplantı Gündemi",            "reports/pazartesi_toplanti.md"),
    ("Gelecek Çalışmalar (cutting-edge)",     "reports/future_work_ai.md"),
    ("Gold-standard Kıyas Tablosu",           "outputs/focus_arch_compare/gold_standard.md"),
    ("Multi-seed Mimari Karşılaştırması",     "outputs/focus_arch_compare/multi_seed_summary.md"),
    ("FocusPoint Karşılaştırma Özeti",        "outputs/focus_arch_compare/summary.md"),
]


def load_api_key() -> str:
    if env := os.environ.get("XAI_API_KEY"):
        return env.strip()
    if KEY_PATH.exists():
        return KEY_PATH.read_text(encoding="utf-8").strip()
    print("error: provide the xAI API key via env XAI_API_KEY or "
          ".grok_key file", file=sys.stderr)
    raise SystemExit(1)


def load_context() -> list[dict]:
    out = []
    for label, rel in CONTEXT_FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"  [skip] missing: {rel}", file=sys.stderr)
            continue
        out.append({
            "label": label,
            "path":  rel,
            "text":  p.read_text(encoding="utf-8"),
        })
        print(f"  [ok] {rel}  ({p.stat().st_size / 1024:.1f} KB)")
    return out


SUGGESTED_QUESTIONS = [
    "Bu projede AI'ı neden kullanıyoruz, klasik yöntemler ne yapamıyor?",
    "Kol A için FNO, U-Net'ten neden 2.7× daha iyi?",
    "Gauge simetrisi nedir, nasıl üç yoldan doğruladınız?",
    "Klasik gold-standard kıyaslamasında hangi yöntemler test edildi?",
    "Z (eksenel) doğruluğu neden klinik için yetersiz?",
    "Faz kuantizasyon adımı neden 5° seçildi?",
    "Veri seti olarak neden OpenBreastUS seçildi?",
    "Sonraki adımlar neler? Transfer learning ne kadar fark eder?",
    "Heatmap-DSNT yaklaşımı baseline'ı geçemedi mi, niye?",
    "İlk denenen 256-faz regresyonu neden çöktü?",
]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIFU Projesi · Danışman Dashboard</title>
<style>
  *  { box-sizing: border-box; }
  :root {
    --c-bg:        #f6f8fb;
    --c-card:      #ffffff;
    --c-border:    #d8e0eb;
    --c-text:      #1a2233;
    --c-muted:     #5b6577;
    --c-accent:    #1f3a5f;
    --c-accent-2:  #ef6c00;
    --c-success:   #2e7d32;
    --c-error:     #c62828;
    --c-msg-user:  #e8f1fb;
    --c-msg-bot:   #fbfbfb;
  }
  body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, sans-serif;
    background: var(--c-bg); color: var(--c-text);
    line-height: 1.55;
  }
  header {
    background: var(--c-accent); color: #fff;
    padding: 18px 28px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .sub { font-size: 13px; opacity: 0.85; margin-top: 3px; }
  .counter {
    background: rgba(255,255,255,0.13);
    padding: 6px 12px; border-radius: 99px; font-size: 13px;
  }
  .counter strong { font-weight: 600; }
  main {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 20px;
    max-width: 1200px;
    margin: 24px auto;
    padding: 0 20px;
  }
  aside {
    background: var(--c-card);
    border: 1px solid var(--c-border);
    border-radius: 8px;
    padding: 18px;
    height: fit-content;
    position: sticky; top: 24px;
  }
  aside h2 {
    margin: 0 0 14px; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--c-muted);
  }
  aside ul { list-style: none; padding: 0; margin: 0; }
  aside li {
    position: relative;
    padding: 12px 14px 12px 36px;
    cursor: pointer;
    border-radius: 8px;
    margin-bottom: 8px;
    color: var(--c-text);
    font-size: 14.5px;
    line-height: 1.5;
    background: #f4f8fc;
    border: 1px solid #e3ecf5;
    border-left: 3px solid transparent;
    transition: all 0.15s ease;
  }
  aside li::before {
    content: counter(item);
    counter-increment: item;
    position: absolute;
    left: 12px; top: 12px;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--c-accent);
    color: #fff;
    font-size: 11px; font-weight: 600;
    display: flex; align-items: center; justify-content: center;
    line-height: 1;
  }
  aside ul { counter-reset: item; }
  aside li:hover {
    background: var(--c-msg-user);
    border-left-color: var(--c-accent);
    transform: translateX(2px);
  }
  aside li:active { transform: translateX(0); }
  aside .info {
    font-size: 12px; color: var(--c-muted);
    margin-top: 20px; padding-top: 16px;
    border-top: 1px solid var(--c-border);
    line-height: 1.6;
  }
  aside .info a { color: var(--c-accent); }
  aside .download-btn {
    width: 100%;
    margin-top: 14px;
    padding: 10px 14px;
    background: #fff;
    border: 1px solid var(--c-border);
    border-radius: 6px;
    color: var(--c-accent);
    font-size: 13px; font-weight: 500;
    cursor: pointer;
    text-align: center;
    transition: all 0.12s;
  }
  aside .download-btn:hover {
    background: var(--c-msg-user);
    border-color: var(--c-accent);
  }
  aside .download-btn:disabled {
    color: var(--c-muted); cursor: not-allowed;
    opacity: 0.6;
  }
  .chat-container {
    background: var(--c-card);
    border: 1px solid var(--c-border);
    border-radius: 8px;
    overflow: hidden;
    display: flex; flex-direction: column;
    min-height: 70vh;
  }
  .messages {
    flex: 1; overflow-y: auto;
    padding: 24px 28px;
    background: var(--c-msg-bot);
  }
  .msg {
    margin-bottom: 18px;
    max-width: 85%;
    padding: 14px 18px;
    border-radius: 12px;
    word-wrap: break-word;
    line-height: 1.6;
  }
  .msg.user {
    background: var(--c-msg-user);
    border: 1px solid var(--c-border);
    margin-left: auto;
  }
  .msg.assistant {
    background: #fff;
    border: 1px solid var(--c-border);
    border-left: 3px solid var(--c-accent);
  }
  .msg.error {
    background: #fdecea;
    border: 1px solid #f5c2c0;
    color: var(--c-error);
  }
  .msg .who {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--c-muted); margin-bottom: 4px; font-weight: 600;
  }
  .msg p { margin: 0 0 10px; }
  .msg p:last-child { margin-bottom: 0; }
  .msg ul, .msg ol { margin: 8px 0; padding-left: 22px; }
  .msg code {
    font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
    font-size: 13px;
    background: #eef2f7; padding: 1px 5px; border-radius: 3px;
  }
  .msg pre {
    background: #1f2937; color: #e5e7eb;
    padding: 12px 14px; border-radius: 6px;
    overflow-x: auto; font-size: 13px;
    margin: 10px 0;
  }
  .msg pre code {
    background: transparent; color: inherit; padding: 0;
  }
  .msg table {
    border-collapse: collapse; margin: 8px 0;
    font-size: 13px;
  }
  .msg th, .msg td {
    border: 1px solid var(--c-border);
    padding: 6px 10px;
  }
  .msg th { background: var(--c-msg-user); font-weight: 600; }
  .msg blockquote {
    margin: 8px 0; padding: 6px 14px;
    border-left: 3px solid var(--c-accent-2);
    color: var(--c-muted); font-style: italic;
  }
  .msg.assistant strong { color: var(--c-accent); }
  .typing {
    display: inline-block;
    color: var(--c-muted); font-style: italic;
  }
  .typing::after {
    content: '...';
    animation: dots 1.2s steps(3, end) infinite;
  }
  @keyframes dots { 0% { content: '.'; } 40% { content: '..'; } 80% { content: '...'; } }
  .composer {
    border-top: 1px solid var(--c-border);
    padding: 16px 20px;
    background: #fff;
    display: flex; gap: 10px;
  }
  .composer textarea {
    flex: 1;
    border: 1px solid var(--c-border);
    border-radius: 6px;
    padding: 10px 14px;
    font-family: inherit; font-size: 14px;
    resize: vertical; min-height: 44px; max-height: 200px;
  }
  .composer textarea:focus {
    outline: none; border-color: var(--c-accent);
    box-shadow: 0 0 0 2px rgba(31, 58, 95, 0.15);
  }
  .composer button {
    background: var(--c-accent); color: #fff;
    border: none; border-radius: 6px;
    padding: 0 18px; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: background 0.12s;
  }
  .composer button:hover:not(:disabled) { background: #15294a; }
  .composer button:disabled { background: #9aa5b8; cursor: not-allowed; }
  .greeting {
    text-align: center; padding: 50px 20px; color: var(--c-muted);
  }
  .greeting h2 { color: var(--c-accent); margin-bottom: 6px; }
  .footer-info {
    text-align: center; font-size: 12px; color: var(--c-muted);
    padding: 12px 20px; border-top: 1px solid var(--c-border);
    background: var(--c-msg-bot);
  }
  @media (max-width: 800px) {
    main { grid-template-columns: 1fr; }
    aside { position: static; }
    header { flex-direction: column; gap: 8px; align-items: flex-start; }
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>HIFU Soft-Tissue · Danışman Dashboard</h1>
    <div class="sub">Proje hakkında soru sorun · Cevaplar yalnızca proje belgelerine dayanır</div>
  </div>
  <div class="counter">Bugün: <strong id="counter-now">0</strong> / 10 soru</div>
</header>

<main>
  <aside>
    <h2>Önerilen Sorular · Tıklayın</h2>
    <ul id="suggested"></ul>
    <button id="download-log" class="download-btn" disabled>
      📥 Konuşma Kaydını İndir (<span id="log-count">0</span>)
    </button>
    <div class="info">
      <strong>Bilgi tabanı:</strong> __KB_COUNT__ doküman, ~__KB_KB__ KB.<br>
      Proje deposu:
      <a href="https://github.com/RsGoksel/Soft-Tissue" target="_blank">github.com/RsGoksel/Soft-Tissue</a><br>
      <em style="opacity:0.75">Günlük 10 soru limiti gece yarısı sıfırlanır.
      İsterseniz oturum sonunda kayıtları indirip iletebilirsiniz.</em>
    </div>
  </aside>

  <section class="chat-container">
    <div class="messages" id="messages">
      <div class="greeting">
        <h2>Hoş geldiniz</h2>
        <p>Soldaki önerilen sorulara tıklayabilir veya kendi sorunuzu aşağıya yazabilirsiniz.</p>
        <p>Tüm cevaplar projenin Türkçe / İngilizce dokümanlarına dayanarak üretilir.</p>
      </div>
    </div>
    <div class="composer">
      <textarea id="input" placeholder="Sorunuzu yazın... (Enter = gönder, Shift+Enter = yeni satır)" rows="1"></textarea>
      <button id="send-btn">Gönder</button>
    </div>
    <div class="footer-info">
      Model: <code>grok-3-latest</code> · API: api.x.ai · Bilgi tabanı yerel
      (RAG yok, tüm dokümanlar her sorguda context'e gönderilir)
    </div>
  </section>
</main>

<script>
const API_KEY = "__API_KEY__";
const API_URL = "https://api.x.ai/v1/chat/completions";
const MODEL   = "grok-3-latest";
const DAILY_LIMIT = 10;

const PROJECT_CONTEXT = __PROJECT_CONTEXT_JSON__;
const SUGGESTED = __SUGGESTED_JSON__;

const SYSTEM_PROMPT = `Sen, RsGoksel/Soft-Tissue isimli HIFU (yüksek yoğunluklu odaklanmış ultrason) tabanlı meme tümörü ablasyon planlama projesi hakkındaki danışman sorularını yanıtlayan bir asistansın.

KURALLAR:
1. **Yalnızca aşağıdaki proje belgelerini bilgi kaynağı olarak kullan.** Belgelerde olmayan bir bilgiye sahipmiş gibi konuşma; bilmediğini söyle.
2. **Türkçe yanıt ver.** Danışman Türk; sinyal işleme uzmanı.
3. Kısa ve net ol. Markdown kullan (tablo, kod bloğu, madde) gerektiğinde.
4. Soru projeyle ilgili değilse kibarca yönlendir.
5. Sayısal sonuçlar verirken hangi belgeden alındığını belirt (örn: "gold_standard.md tablosuna göre...").
6. İlk yaklaşımın neden çöktüğü, gauge serbestliği, faz kuantizasyon, gold-standard kıyas — projenin en güçlü tartışma noktaları; bunlarda detaylı cevap ver.

PROJE BELGELERİ:
${PROJECT_CONTEXT.map(d => `\n--- DOKÜMAN: ${d.label} (${d.path}) ---\n${d.text}\n`).join("\n")}
--- BELGELER SONU ---`;

const messages = [];
const $ = (id) => document.getElementById(id);

function todayKey() {
  return new Date().toISOString().slice(0, 10);
}
function readCounter() {
  try {
    const obj = JSON.parse(localStorage.getItem("hoca_counter") || "{}");
    if (obj.date !== todayKey()) return { date: todayKey(), count: 0 };
    return obj;
  } catch (e) { return { date: todayKey(), count: 0 }; }
}
function writeCounter(c) {
  localStorage.setItem("hoca_counter", JSON.stringify(c));
}
function refreshCounter() {
  const c = readCounter();
  $("counter-now").textContent = c.count;
  $("send-btn").disabled = c.count >= DAILY_LIMIT;
  if (c.count >= DAILY_LIMIT) {
    $("send-btn").textContent = "Limit doldu";
  }
}

// --- Q&A logging ---------------------------------------------------------
// Every question and answer is timestamped and stored in localStorage so
// the operator can later download the transcript for system improvement.
function readLog() {
  try { return JSON.parse(localStorage.getItem("hoca_qa_log") || "[]"); }
  catch (e) { return []; }
}
function appendLog(entry) {
  const log = readLog();
  log.push(entry);
  localStorage.setItem("hoca_qa_log", JSON.stringify(log));
  refreshLogButton();
}
function refreshLogButton() {
  const log = readLog();
  $("log-count").textContent = log.length;
  $("download-log").disabled = log.length === 0;
}
function downloadLog() {
  const log = readLog();
  if (log.length === 0) return;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const payload = {
    exported_at: new Date().toISOString(),
    user_agent:  navigator.userAgent,
    daily_count: readCounter().count,
    entries:     log,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                        { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `hoca_qa_log_${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// minimal-but-usable markdown renderer
function md(text) {
  text = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // code blocks
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code>${code.trim()}</code></pre>`);
  // inline code
  text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  // headings
  text = text.replace(/^###### (.+)$/gm, '<h6>$1</h6>');
  text = text.replace(/^##### (.+)$/gm, '<h5>$1</h5>');
  text = text.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // bold + italic
  text = text.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
  // tables
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\|.*\|$/.test(line) && i + 1 < lines.length && /^\|[\s\-:|]+\|$/.test(lines[i+1])) {
      const headers = line.split("|").slice(1, -1).map(s => s.trim());
      const rows = [];
      i += 2;
      while (i < lines.length && /^\|.*\|$/.test(lines[i])) {
        rows.push(lines[i].split("|").slice(1, -1).map(s => s.trim()));
        i++;
      }
      out.push("<table><thead><tr>" +
        headers.map(h => `<th>${h}</th>`).join("") +
        "</tr></thead><tbody>" +
        rows.map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>");
      continue;
    }
    out.push(line);
    i++;
  }
  text = out.join("\n");
  // unordered lists
  text = text.replace(/(?:^- .+(?:\n|$))+/gm, (block) => {
    const items = block.trim().split(/\n/).map(l => l.replace(/^- /, "").trim());
    return "<ul>" + items.map(it => `<li>${it}</li>`).join("") + "</ul>";
  });
  // numbered lists
  text = text.replace(/(?:^\d+\. .+(?:\n|$))+/gm, (block) => {
    const items = block.trim().split(/\n/).map(l => l.replace(/^\d+\.\s+/, "").trim());
    return "<ol>" + items.map(it => `<li>${it}</li>`).join("") + "</ol>";
  });
  // blockquotes
  text = text.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // paragraphs (single newlines -> <br>, blank lines split)
  text = text.split(/\n\n+/).map(p => {
    if (/^<(h[1-6]|ul|ol|table|pre|blockquote)/.test(p.trim())) return p;
    return "<p>" + p.replace(/\n/g, "<br>") + "</p>";
  }).join("\n");
  return text;
}

function appendMsg(role, text) {
  const m = $("messages");
  // remove greeting on first append
  const greet = m.querySelector(".greeting");
  if (greet) greet.remove();
  const div = document.createElement("div");
  div.className = "msg " + role;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "Soru" : (role === "error" ? "Hata" : "Asistan");
  div.appendChild(who);
  const body = document.createElement("div");
  body.innerHTML = role === "user" ? text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\n/g,"<br>") : md(text);
  div.appendChild(body);
  m.appendChild(div);
  m.scrollTop = m.scrollHeight;
  return div;
}

function appendTyping() {
  const m = $("messages");
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.id = "typing";
  div.innerHTML = '<div class="who">Asistan</div><span class="typing">Yanıt oluşturuluyor</span>';
  m.appendChild(div);
  m.scrollTop = m.scrollHeight;
  return div;
}

async function send() {
  const input = $("input");
  const text = input.value.trim();
  if (!text) return;
  const c = readCounter();
  if (c.count >= DAILY_LIMIT) return;

  const tsQ = new Date();
  appendLog({
    ts: tsQ.toISOString(),
    role: "user",
    content: text,
    daily_index: c.count + 1,
  });

  appendMsg("user", text);
  messages.push({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";
  $("send-btn").disabled = true;
  const typing = appendTyping();
  const tStart = performance.now();

  try {
    const r = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0.3,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          ...messages,
        ],
      }),
    });
    typing.remove();
    if (!r.ok) {
      const err = await r.text();
      appendMsg("error", `API hatası (${r.status}): ${err.slice(0, 400)}`);
      appendLog({
        ts: new Date().toISOString(),
        role: "error",
        status_code: r.status,
        content: err.slice(0, 400),
      });
      $("send-btn").disabled = false;
      return;
    }
    const data = await r.json();
    const reply = data.choices?.[0]?.message?.content || "(boş yanıt)";
    const latency_ms = Math.round(performance.now() - tStart);
    appendMsg("assistant", reply);
    messages.push({ role: "assistant", content: reply });
    appendLog({
      ts: new Date().toISOString(),
      role: "assistant",
      content: reply,
      latency_ms: latency_ms,
      model: MODEL,
      usage: data.usage || null,
    });
    c.count += 1;
    writeCounter(c);
    refreshCounter();
  } catch (e) {
    typing.remove();
    appendMsg("error", `Bağlantı hatası: ${e.message}`);
    appendLog({
      ts: new Date().toISOString(),
      role: "error",
      content: `Bağlantı hatası: ${e.message}`,
    });
    $("send-btn").disabled = false;
  }
}

function setupSuggested() {
  const ul = $("suggested");
  SUGGESTED.forEach(q => {
    const li = document.createElement("li");
    li.textContent = q;
    li.onclick = () => {
      $("input").value = q;
      $("input").focus();
    };
    ul.appendChild(li);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  refreshCounter();
  refreshLogButton();
  setupSuggested();
  $("send-btn").addEventListener("click", send);
  $("download-log").addEventListener("click", downloadLog);
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  $("input").addEventListener("input", (e) => {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  });
});
</script>
</body>
</html>
"""


def main() -> None:
    api_key = load_api_key()
    print(f"[gen] API key loaded ({len(api_key)} chars)")

    docs = load_context()
    total_chars = sum(len(d["text"]) for d in docs)
    print(f"[gen] embedding {len(docs)} docs, {total_chars / 1024:.1f} KB total")

    html = HTML_TEMPLATE
    html = html.replace("__API_KEY__",            api_key)
    html = html.replace("__PROJECT_CONTEXT_JSON__", json.dumps(docs, ensure_ascii=False))
    html = html.replace("__SUGGESTED_JSON__",     json.dumps(SUGGESTED_QUESTIONS, ensure_ascii=False))
    html = html.replace("__KB_COUNT__",           str(len(docs)))
    html = html.replace("__KB_KB__",              f"{total_chars / 1024:.0f}")

    OUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[gen] wrote {OUT_PATH}  ({size_kb:.0f} KB)")
    print(f"[gen] open in any modern browser; share the file with the advisor")
    print(f"[gen] DO NOT commit — already in .gitignore")


if __name__ == "__main__":
    main()
