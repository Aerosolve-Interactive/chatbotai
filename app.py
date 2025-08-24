# app.py — Slate API v0.4.1
# - Plain-English math (solve/roots/derivative/integral/simplify/“what is …”)
# - Writing coach (shorten/explain/rewrite). Uses OpenAI if OPENAI_API_KEY is set; else safe fallback.
# - Safe Python code runner (no network/shell; CPU/RAM/time limits)
# - Never 500s (all exceptions returned as JSON)
# Endpoints: GET / , GET /health , POST /chat , GET /widget , GET /w , GET /widget_debug

from typing import Literal, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import os, re, tempfile, subprocess, sys

# ========== Math stack ==========
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# ---- JSON-safe stringify for SymPy/containers (prevents 500s) ----
def _stringify(obj):
    """Recursively convert SymPy objects (and dict KEYS) to plain strings; turn sets/tuples into lists."""
    try:
        import sympy as _sp
        if isinstance(obj, (_sp.Basic,)):
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, set):
        return [_stringify(v) for v in obj]
    return obj

app = FastAPI(title="Slate API", version="0.4.1")

# CORS: keep "*" for MVP; lock to your Wix domain later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Math (plain English) tool
# =========================
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # "5x" -> 5*x , "2(x+1)" -> 2*(x+1)
    convert_xor,                          # "^"  -> "**"
)

def sp_parse(s: str):
    s = (s or "").strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("ln", "log")  # SymPy uses log() for natural log
    return parse_expr(s, transformations=TRANSFORMS)

def english_to_math(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    low = raw.lower()

    # Solve explicit equation when user hints solving
    if "=" in raw and any(k in low for k in ("solve", "for x", "for y", "root", "roots", "zeros", "zeroes")):
        L, R = raw.split("=", 1)
        return {"op": "solve", "left": L, "right": R}

    # "roots of f(x)"
    m = re.search(r"(?:roots?|zeros?|zeroes?)\s+of\s+(.+)", raw, flags=re.I)
    if m: return {"op": "solve", "left": m.group(1), "right": "0"}

    # "solve ..." (assume = 0)
    m = re.search(r"\bsolve\b\s+(.+)", raw, flags=re.I)
    if m: return {"op": "solve", "left": m.group(1), "right": "0"}

    # "solve for x: 2x+3=7"
    m = re.search(r"\bsolve\s+for\s+([a-z])\s*:\s*(.+)", raw, flags=re.I)
    if m:
        var, eq = m.group(1), m.group(2)
        if "=" in eq: L, R = eq.split("=", 1)
        else: L, R = eq, "0"
        return {"op": "solve", "left": L, "right": R, "var": var}

    # Derivative
    m = re.search(r"\bd\/dx\b\s+(.+)", raw, flags=re.I)
    if m: return {"op": "diff", "expr": m.group(1)}
    m = re.search(r"(?:derivative of|differentiate)\s+(.+)", raw, flags=re.I)
    if m: return {"op": "diff", "expr": m.group(1)}

    # Integral
    m = re.search(r"(?:integral of|antiderivative of|integrate)\s+(.+)", raw, flags=re.I)
    if m: return {"op": "integrate", "expr": m.group(1)}

    # Simplify
    m = re.search(r"\bsimplify\b\s+(.+)", raw, flags=re.I)
    if m: return {"op": "simplify", "expr": m.group(1)}

    # Evaluate / compute / what's ...
    if any(k in low for k in ("what is", "what's", "whats", "value of", "compute", "evaluate")):
        expr = re.sub(r".*?(what is|what's|whats|value of|compute|evaluate)\s*", "", raw, flags=re.I)
        return {"op": "eval", "expr": expr}

    # Fallbacks
    if "=" in raw:
        L, R = raw.split("=", 1)
        return {"op": "solve", "left": L, "right": R}
    return {"op": "eval", "expr": raw}

def math_tool(text: str) -> Dict[str, Any]:
    try:
        spec = english_to_math(text)
        op = spec.get("op")
        if op == "solve":
            L, R = sp_parse(spec["left"]), sp_parse(spec["right"])
            eq = sp.Eq(L, R)
            syms = sorted(eq.free_symbols, key=lambda s: s.name)
            var = syms[0] if syms else sp.symbols(spec.get("var", "x"))
            sol = sp.solve(eq, var, dict=True)   # SymPy objects
            return {
                "type": "solve",
                "equation": str(eq),
                "symbol": str(var),
                "solution": _stringify(sol),   # JSON-safe
            }
        if op == "diff":
            x = sp.symbols("x")
            expr = sp_parse(spec["expr"])
            return {"type": "diff", "expr": str(expr), "d/dx": str(sp.diff(expr, x))}
        if op == "integrate":
            x = sp.symbols("x")
            expr = sp_parse(spec["expr"])
            return {"type": "integrate", "expr": str(expr), "∫dx": str(sp.integrate(expr, x))}
        if op == "simplify":
            expr = sp_parse(spec["expr"])
            return {"type": "simplify", "expr": str(expr), "simplified": str(sp.simplify(expr))}
        if op == "eval":
            expr = sp_parse(spec["expr"])
            if not expr.free_symbols:
                return {"type": "eval", "expr": str(expr), "value": str(sp.N(expr))}
            return {"type": "eval", "expr": str(expr), "simplified": str(sp.simplify(expr))}
        return {"error": "Unrecognized math intent."}
    except Exception as e:
        return {"error": f"Math parse/solve failed: {e}"}

# ==================
# Writing coach tool
# ==================
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))
    _model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _OPENAI_AVAILABLE = False

WRITING_SYSTEM = (
    "You are a strict but helpful writing tutor for grades 7–12. "
    "Follow the instruction (shorten, explain, rewrite) while preserving meaning and citations. "
    "Prefer clarity, concision, active voice. Return ONLY the revised text."
)

def llm_rewrite(prompt: str) -> Optional[str]:
    if not _OPENAI_AVAILABLE:
        return None
    try:
        resp = _client.chat.completions.create(
            model=_model,
            messages=[{"role":"system","content":WRITING_SYSTEM},{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def detect_writing_task(text: str) -> Dict[str, Any]:
    low = (text or "").lower()
    if "shorten" in low or "condense" in low or "make it shorter" in low:
        m = re.search(r"(?:shorten|condense).{0,40}?(\d+)\s*(?:words|w)", low)
        return {"task":"shorten", "target": int(m.group(1)) if m else None}
    if "explain" in low or "simplify" in low or "make it simpler" in low:
        m = re.search(r"(?:grade|reading level)\s*(\d+)", low)
        return {"task":"explain", "grade": int(m.group(1)) if m else 9}
    if any(k in low for k in ["rewrite","rephrase","improve","fix grammar"]):
        m = re.search(r"(?:tone|style)\s*(formal|casual|academic|concise)", low)
        return {"task":"rewrite", "tone": m.group(1) if m else "concise academic"}
    return {"task":"rewrite", "tone":"concise academic"}

def fallback_shorten(text: str, target: Optional[int]) -> str:
    words = text.split()
    if target and target < len(words):
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        out, count = [], 0
        for s in sents:
            w = len(s.split())
            if count + w <= target or not out:
                out.append(s); count += w
            else:
                break
        return " ".join(out)
    return re.sub(r"\s+", " ", text).strip()

def fallback_explain(text: str, grade: int) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    simplified = []
    for s in sents:
        s = re.sub(r"\s+", " ", s.strip())
        if len(s) > 220:
            parts = re.split(r"(;|,|\band\b)", s)
            s = " ".join(parts[:max(2, len(parts)//2)])
        simplified.append(s)
    return " ".join(simplified)

def fallback_rewrite(text: str, tone: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\bi\b", "I", t)
    return t

def writing_tool(text: str) -> Dict[str, Any]:
    try:
        task = detect_writing_task(text)
        m = re.search(r":\s*(.+)$", text, flags=re.S)
        content = m.group(1) if m else text
        if task["task"] == "shorten" and task.get("target"):
            prompt = f"Shorten to about {task['target']} words:\n\n{content}"
        elif task["task"] == "explain":
            prompt = f"Explain at about grade {task.get('grade',9)} level. Keep key points:\n\n{content}"
        else:
            prompt = f"Rewrite in a {task.get('tone','concise academic')} tone, improve clarity/grammar:\n\n{content}"
        out = llm_rewrite(prompt)
        if out:
            return {"type": task["task"], "engine":"llm", "output": out}
        # Fallbacks (no API key)
        if task["task"] == "shorten":
            return {"type":"shorten","engine":"fallback","output": fallback_shorten(content, task.get("target"))}
        if task["task"] == "explain":
            return {"type":"explain","engine":"fallback","output": fallback_explain(content, task.get("grade",9))}
        return {"type":"rewrite","engine":"fallback","output": fallback_rewrite(content, task.get("tone","concise academic"))}
    except Exception as e:
        return {"error": f"Writing tool failed: {e}"}

# ===================
# Code tool + runner
# ===================
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m: return m.group(1)
    if text.lower().startswith("run:"):
        return text.split(":",1)[1]
    return text

def safe_run_python(code: str) -> Dict[str, Any]:
    try:
        # syntax check
        import ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"syntax":"error","detail": f"{e.msg} at {e.lineno}:{e.offset}"}

        with tempfile.TemporaryDirectory() as td:
            main_py = os.path.join(td, "main.py")
            wrapper_py = os.path.join(td, "wrapper.py")
            open(main_py, "w", encoding="utf-8").write(code)

            wrapper = r"""
import sys, os, runpy
# resource limits (Linux-only)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2,2))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
except Exception:
    pass
# block networking and shell-outs
import socket, subprocess
def _block(*a, **k): raise RuntimeError("disabled")
socket.socket = _block
subprocess.Popen = _block
os.system = _block
# run user code
runpy.run_path("main.py", run_name="__main__")
"""
            open(wrapper_py, "w", encoding="utf-8").write(wrapper)

            proc = subprocess.run(
                [sys.executable, "-I", "wrapper.py"],
                cwd=td, capture_output=True, timeout=3, text=True
            )
            return {
                "syntax":"ok",
                "ran": True,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-8000:],  # cap size
                "stderr": proc.stderr[-8000:],
                "timeout": False
            }
    except subprocess.TimeoutExpired:
        return {"syntax":"ok","ran": True, "timeout": True, "stdout":"", "stderr":"Timed out after 3s"}
    except Exception as e:
        return {"error": f"Runner failed: {e}"}

def code_tool(text: str) -> Dict[str, Any]:
    try:
        code = extract_code(text)
        return safe_run_python(code)
    except Exception as e:
        return {"error": f"Code tool failed: {e}"}

# =========
# API layer
# =========
class ChatIn(BaseModel):
    mode: Literal["auto","math","write","code"] = "auto"
    text: str

@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <html><body style="font-family:ui-sans-serif;background:#0b0b0c;color:#e7e7ea">
      <h3>Slate API v0.4.1</h3>
      <ul>
        <li><a href="/health" style="color:#8b5cf6">/health</a></li>
        <li><a href="/docs" style="color:#8b5cf6">/docs</a> (test POST /chat here)</li>
        <li><a href="/widget?v=2" style="color:#8b5cf6">/widget</a> · <a href="/w?v=1" style="color:#8b5cf6">/w</a></li>
      </ul>
    </body></html>
    """

# -------------------------------
# Hardened widget at /widget
# -------------------------------
WIDGET_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Slate — Study Assistant</title>
<style>
  :root{--bg:#0b0b0c;--panel:#121214;--panel-2:#17171a;--text:#e7e7ea;--muted:#a1a1aa;--accent:#8b5cf6}
  *{box-sizing:border-box} html,body{height:100%} body{margin:0;font-family:ui-sans-serif,system-ui,Inter;background:var(--bg);color:var(--text)}
  .wrap{display:flex;flex-direction:column;height:100%;padding:16px}
  .card{flex:1;display:flex;flex-direction:column;background:var(--panel);border:1px solid #222226;border-radius:16px;overflow:hidden}
  .header{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid #222226;background:#17171a}
  .brand{display:flex;gap:10px;align-items:center;font-weight:700}.dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px var(--accent)}
  .seg{display:inline-flex;background:#0f0f12;padding:4px;border-radius:999px;border:1px solid #1f1f24}
  .seg button{background:transparent;border:0;color:var(--muted);padding:6px 12px;border-radius:999px;cursor:pointer;font-size:12px}
  .seg button.active{background:#17171a;color:var(--text);border:1px solid #2a2a31}
  .chat{flex:1;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
  .msg{max-width:88%;padding:10px 12px;border-radius:12px;font-size:14px;white-space:pre-wrap;border:1px solid #24242a}
  .msg.user{align-self:flex-end;background:#101014;border-color:#2a2a31}.msg.bot{align-self:flex-start;background:#0f0f12}
  .footer{display:flex;gap:8px;padding:12px;border-top:1px solid #222226;background:#17171a}
  textarea{flex:1;resize:none;background:#0f0f12;color:#e7e7ea;border:1px solid #26262c;border-radius:12px;padding:10px 12px;min-height:48px;max-height:160px}
  .send{background:var(--accent);color:#fff;border:0;border-radius:12px;padding:0 16px;cursor:pointer;font-weight:600}
  .hint{color:var(--muted);font-size:12px;padding:0 14px 12px}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <div class="brand"><span class="dot"></span><span>Slate</span><span style="color:#a1a1aa;font-size:12px">· math · code · writing</span></div>
        <div id="sl-seg" class="seg" role="tablist" aria-label="Mode">
          <button class="active" data-mode="auto">Auto</button>
          <button data-mode="math">Math</button>
          <button data-mode="code">Code</button>
          <button data-mode="write">Writing</button>
        </div>
      </div>
      <div id="sl-chat" class="chat" aria-live="polite"></div>
      <div class="hint">Enter to send • Shift+Enter for newline</div>
      <div class="footer">
        <textarea id="sl-input" placeholder="Ask in plain English. e.g., ‘Find the roots of x^2 - 5x + 6’ or ‘Shorten to 120 words: …’"></textarea>
        <button id="sl-send" class="send" type="button">Send</button>
      </div>
    </div>
  </div>

<script>
(function(){
  const BACKEND_URL = location.origin; // same-origin = no CORS issues
  const chat  = document.getElementById("sl-chat");
  const input = document.getElementById("sl-input");
  const send  = document.getElementById("sl-send");
  const seg   = document.getElementById("sl-seg");
  let mode="auto", sending=false, warmed=false;

  seg.querySelectorAll("button").forEach(b=>{
    b.addEventListener("click", ()=>{
      seg.querySelectorAll("button").forEach(x=>x.classList.remove("active"));
      b.classList.add("active"); mode=b.dataset.mode;
    });
  });

  function add(text, who="bot"){
    const d=document.createElement("div");
    d.className="msg "+(who==="user"?"user":"bot");
    d.textContent=text; chat.appendChild(d); chat.scrollTop=chat.scrollHeight;
  }

  function render(obj){
    if(!obj) return add("No response.","bot");
    if(obj.error) return add("Error: "+obj.error,"bot");
    const {mode, result}=obj;
    if(mode==="math"){
      if(result.error) return add("Math error: "+result.error,"bot");
      if(result.type==="eval" && result.value!==undefined) return add("Value: "+result.value,"bot");
      if(result.type==="simplify") return add(`Simplified ${result.expr} → ${result.simplified}`,"bot");
      if(result.type==="diff") return add(`d/dx of ${result.expr} = ${result["d/dx"]}`,"bot");
      if(result.type==="integrate") return add(`∫ ${result.expr} dx = ${result["∫dx"]}`,"bot");
      if(result.type==="solve") return add(`Equation: ${result.equation}\nSymbol: ${result.symbol}\nSolution: ${JSON.stringify(result.solution,null,2)}`,"bot");
      return add(JSON.stringify(result),"bot");
    }
    if(mode==="code"){
      if(result.error) return add("Code error: "+result.error,"bot");
      if(result.syntax==="error") return add("Syntax error: "+result.detail,"bot");
      if(result.timeout) return add("Your code timed out (3s limit).","bot");
      let out="Ran ✅";
      if(result.stdout) out+="\nstdout:\n"+result.stdout.trim();
      if(result.stderr) out+="\nstderr:\n"+result.stderr.trim();
      return add(out,"bot");
    }
    // write
    if(result.error) return add("Writing error: "+result.error,"bot");
    return add(result.output || JSON.stringify(result),"bot");
  }

  async function go(){
    if(sending) return;
    const text=(input.value||"").trim(); if(!text) return;
    add(text,"user"); input.value=""; sending=true;

    let hint=setTimeout(()=>add("…waking the server…","bot"),1200);
    try{
      const r=await fetch(BACKEND_URL+"/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode,text})});
      const j=await r.json(); clearTimeout(hint); render(j);
      if(!warmed){ warmed=true; setTimeout(()=>{ fetch(BACKEND_URL+"/health",{method:"HEAD"}).catch(()=>{}); }, 120000); }
    }catch(e){ clearTimeout(hint); add("Network error: "+e.message,"bot"); }
    finally{ sending=false; }
  }

  send.addEventListener("click", go);
  input.addEventListener("keydown", (e)=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); go(); } });

  add("Hi! I’m Slate. Try: ‘what is 8*8’.","bot");
})();
</script>
</body></html>"""

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return HTMLResponse(content=WIDGET_HTML, headers={"Cache-Control": "no-store"})

# -------------------------------
# Ultra-minimal widget at /w
# -------------------------------
NEW_WIDGET_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Slate — Minimal Widget</title>
<style>
  :root{--bg:#0b0b0c;--panel:#121214;--panel2:#17171a;--text:#e7e7ea;--muted:#a1a1aa;--accent:#8b5cf6}
  *{box-sizing:border-box} html,body{height:100%} body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,Inter}
  .wrap{display:flex;flex-direction:column;gap:12px;height:100%;padding:16px}
  .bar{display:flex;gap:8px;align-items:center}
  .tab{border:1px solid #2a2a31;background:#0f0f12;color:var(--muted);padding:6px 10px;border-radius:999px;cursor:pointer;font-size:12px}
  .tab.active{color:#fff;background:var(--panel2)}
  .box{flex:1;overflow:auto;background:var(--panel);border:1px solid #222226;border-radius:12px;padding:12px}
  .msg{max-width:85%;padding:8px 10px;border-radius:10px;border:1px solid #2a2a31;margin:6px 0;white-space:pre-wrap}
  .u{margin-left:auto;background:#101014}
  .b{margin-right:auto;background:#0f0f12}
  .input{display:flex;gap:8px}
  textarea{flex:1;background:#0f0f12;color:var(--text);border:1px solid #26262c;border-radius:10px;padding:10px;min-height:48px}
  button{background:var(--accent);color:#fff;border:0;border-radius:10px;padding:0 14px;cursor:pointer;font-weight:600}
</style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <strong>Slate</strong>
      <div style="flex:1"></div>
      <button class="tab active" data-w-mode="auto">Auto</button>
      <button class="tab" data-w-mode="math">Math</button>
      <button class="tab" data-w-mode="code">Code</button>
      <button class="tab" data-w-mode="write">Writing</button>
    </div>
    <div id="w-chat" class="box" aria-live="polite"></div>
    <div class="input">
      <textarea id="w-input" placeholder="Try: what is 8*8  ·  or  find the roots of x^2-5x+6"></textarea>
      <button id="w-send" type="button">Send</button>
    </div>
  </div>

<script>
(function(){
  const BACKEND_URL = location.origin;  // same-origin
  const chat  = document.getElementById("w-chat");
  const input = document.getElementById("w-input");
  const send  = document.getElementById("w-send");
  const tabs  = Array.from(document.querySelectorAll("[data-w-mode]"));
  let mode = "auto", sending = false, warmed = false;

  tabs.forEach(btn => btn.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    mode = btn.dataset.wMode;  // "math"|"code"|"write"|"auto"
  }));

  function add(text, who){
    const d = document.createElement("div");
    d.className = "msg " + (who === "user" ? "u" : "b");
    d.textContent = text;
    chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
  }

  function render(obj){
    if(!obj) return add("No response.","bot");
    if(obj.error) return add("Error: " + obj.error, "bot");
    const {mode, result} = obj;
    if(mode === "math"){
      if(result.error) return add("Math error: " + result.error, "bot");
      if(result.type === "eval" && result.value !== undefined) return add("Value: " + result.value, "bot");
      return add(JSON.stringify(result), "bot");
    }
    if(mode === "code"){
      if(result.error) return add("Code error: " + result.error, "bot");
      let out = "Ran ✅";
      if(result.stdout) out += "\\nstdout:\\n" + result.stdout.trim();
      if(result.stderr) out += "\\nstderr:\\n" + result.stderr.trim();
      return add(out, "bot");
    }
    if(result.error) return add("Writing error: " + result.error, "bot");
    return add(result.output || JSON.stringify(result), "bot");
  }

  async function go(){
    if(sending) return;
    const text = (input.value || "").trim();
    if(!text) return;
    add(text, "user");
    input.value = "";
    sending = true;

    let nudge = setTimeout(()=>add("…waking the server…","bot"), 1200);
    try{
      const r = await fetch(BACKEND_URL + "/chat", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ mode, text })
      });
      const j = await r.json();
      clearTimeout(nudge); render(j);

      if(!warmed){ warmed = true; setTimeout(()=>{ fetch(BACKEND_URL + "/health", {method:"HEAD"}).catch(()=>{}); }, 120000); }
    }catch(e){
      clearTimeout(nudge); add("Network error: " + e.message, "bot");
    }finally{ sending = false; }
  }

  send.addEventListener("click", go);
  input.addEventListener("keydown", e => { if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); go(); } });

  add("Hi! I’m Slate. Try: “what is 8*8”.", "bot");
})();
</script>
</body></html>"""

@app.get("/w", response_class=HTMLResponse)
def widget_min():
    return HTMLResponse(content=NEW_WIDGET_HTML, headers={"Cache-Control": "no-store"})

# --------------------------------
# Diagnostic page at /widget_debug
# --------------------------------
DIAG_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Slate Widget Debug</title>
<style>
  body{background:#0b0b0c;color:#e7e7ea;font-family:ui-sans-serif;margin:0;padding:16px}
  button{background:#8b5cf6;color:#fff;border:0;border-radius:10px;padding:8px 12px;margin-right:8px;cursor:pointer}
  pre{white-space:pre-wrap;background:#121214;border:1px solid #222226;border-radius:12px;padding:12px;margin-top:12px}
</style>
</head>
<body>
  <h3>Widget Debug</h3>
  <div>
    <button id="b1">Toggle Mode</button>
    <button id="b2">Send “what is 2+2”</button>
    <button id="b3">Ping /health (HEAD)</button>
  </div>
  <pre id="log">Booting…</pre>

<script>
  const BACKEND_URL = location.origin;  // same-origin
  const logEl = document.getElementById('log');
  const log = (m) => { logEl.textContent += "\\n" + m; };

  window.addEventListener('error', e => { log("JS ERROR: " + e.message); });

  log("Boot OK (JS loaded). UserAgent=" + navigator.userAgent);

  let mode = "auto";
  document.getElementById('b1').addEventListener('click', () => {
    mode = (mode === "auto" ? "math" : "auto");
    log("Mode now: " + mode);
  });

  document.getElementById('b2').addEventListener('click', async () => {
    log("POST /chat …");
    try {
      const r = await fetch(BACKEND_URL + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, text: "what is 2+2" })
      });
      log("Status: " + r.status);
      const j = await r.json();
      log("Body: " + JSON.stringify(j));
    } catch (e) {
      log("Network error: " + e.message);
    }
  });

  document.getElementById('b3').addEventListener('click', async () => {
    log("HEAD /health …");
    try {
      const r = await fetch(BACKEND_URL + "/health", { method: "HEAD", cache: "no-store" });
      log("HEAD status: " + r.status);
    } catch (e) {
      log("Network error: " + e.message);
    }
  });
</script>
</body></html>"""

@app.get("/widget_debug", response_class=HTMLResponse)
def widget_debug():
    return HTMLResponse(content=DIAG_HTML, headers={"Cache-Control": "no-store"})

# ----------------
# Misc endpoints
# ----------------
@app.get("/favicon.ico")
def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/health")
def health():
    return {"status": "ok"}

def route(mode: str, text: str):
    t = (text or "").strip()
    if not t: return {"error":"Empty prompt."}
    low = t.lower()
    is_mathy = any(k in low for k in [
        "solve","root","zero","derivative","differentiate","d/dx","integral","integrate",
        "simplify","value of","compute","evaluate","what is","what's","whats","=","^"
    ])
    is_codey = any(k in low for k in ["```", "def ", "class ", "run:", "python"])
    if mode == "math" or (mode == "auto" and is_mathy):
        return {"mode":"math","result": math_tool(t)}
    if mode == "code" or (mode == "auto" and is_codey):
        return {"mode":"code","result": code_tool(t)}
    return {"mode":"write","result": writing_tool(t)}

@app.post("/chat")
def chat(payload: ChatIn):
    try:
        return route(payload.mode, payload.text)
    except Exception as e:
        return {"error": f"Server error: {e}"}
