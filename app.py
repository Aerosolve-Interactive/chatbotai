# app.py — Slate API v0.6.0
# Auto intent (math / code-generate / code-run / writing / general-knowledge)
# Math: SymPy; Writing: shorten/explain/rewrite/lengthen (LLM if key, else fallback)
# Code: generate (LLM or fallback) OR run (safe Python sandbox)
# General knowledge: uses LLM if key; else free web (DuckDuckGo + Wikipedia)
# Endpoints: GET / , GET /health , POST /chat , GET /widget (302→/w) , GET /w , GET /widget_debug

from typing import Literal, Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from pydantic import BaseModel
import os, re, tempfile, subprocess, sys, json

# stdlib HTTP for zero extra deps
import urllib.request, urllib.parse, ssl
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = True
_ssl_ctx.verify_mode = ssl.CERT_REQUIRED

# ========== Math stack ==========
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

def _stringify(obj):
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

app = FastAPI(title="Slate API", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Math (plain English) tool
# =========================
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def sp_parse(s: str):
    s = (s or "").strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("ln", "log")
    return parse_expr(s, transformations=TRANSFORMS)

def english_to_math(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    low = raw.lower()

    if "=" in raw and any(k in low for k in ("solve","for x","for y","root","roots","zeros","zeroes")):
        L, R = raw.split("=", 1); return {"op":"solve","left":L,"right":R}
    m = re.search(r"(?:roots?|zeros?|zeroes?)\s+of\s+(.+)", raw, flags=re.I)
    if m: return {"op":"solve","left":m.group(1),"right":"0"}
    m = re.search(r"\bsolve\b\s+(.+)", raw, flags=re.I)
    if m: return {"op":"solve","left":m.group(1),"right":"0"}
    m = re.search(r"\bsolve\s+for\s+([a-z])\s*:\s*(.+)", raw, flags=re.I)
    if m:
        var, eq = m.group(1), m.group(2)
        if "=" in eq: L, R = eq.split("=", 1)
        else: L, R = eq, "0"
        return {"op":"solve","left":L,"right":R,"var":var}
    m = re.search(r"\bd\/dx\b\s+(.+)", raw, flags=re.I)
    if m: return {"op":"diff","expr":m.group(1)}
    m = re.search(r"(?:derivative of|differentiate)\s+(.+)", raw, flags=re.I)
    if m: return {"op":"diff","expr":m.group(1)}
    m = re.search(r"(?:integral of|antiderivative of|integrate)\s+(.+)", raw, flags=re.I)
    if m: return {"op":"integrate","expr":m.group(1)}
    m = re.search(r"\bsimplify\b\s+(.+)", raw, flags=re.I)
    if m: return {"op":"simplify","expr":m.group(1)}
    if any(k in low for k in ("what is","what's","whats","value of","compute","evaluate")):
        expr = re.sub(r".*?(what is|what's|whats|value of|compute|evaluate)\s*", "", raw, flags=re.I)
        return {"op":"eval","expr":expr}
    if "=" in raw:
        L, R = raw.split("=", 1); return {"op":"solve","left":L,"right":R}
    return {"op":"eval","expr":raw}

def math_tool(text: str) -> Dict[str, Any]:
    try:
        spec = english_to_math(text); op = spec.get("op")
        if op == "solve":
            L, R = sp_parse(spec["left"]), sp_parse(spec["right"])
            eq = sp.Eq(L, R)
            syms = sorted(eq.free_symbols, key=lambda s: s.name)
            var = syms[0] if syms else sp.symbols(spec.get("var","x"))
            sol = sp.solve(eq, var, dict=True)
            return {"type":"solve","equation":str(eq),"symbol":str(var),"solution":_stringify(sol)}
        if op == "diff":
            x = sp.symbols("x"); expr = sp_parse(spec["expr"])
            return {"type":"diff","expr":str(expr),"d/dx":str(sp.diff(expr,x))}
        if op == "integrate":
            x = sp.symbols("x"); expr = sp_parse(spec["expr"])
            return {"type":"integrate","expr":str(expr),"∫dx":str(sp.integrate(expr,x))}
        if op == "simplify":
            expr = sp_parse(spec["expr"])
            return {"type":"simplify","expr":str(expr),"simplified":str(sp.simplify(expr))}
        if op == "eval":
            expr = sp_parse(spec["expr"])
            if not expr.free_symbols:
                return {"type":"eval","expr":str(expr),"value":str(sp.N(expr))}
            return {"type":"eval","expr":str(expr),"simplified":str(sp.simplify(expr))}
        return {"error":"Unrecognized math intent."}
    except Exception as e:
        return {"error": f"Math parse/solve failed: {e}"}

# ==================
# Writing tools
# ==================
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))
    _model = os.getenv("OPENAI_MODEL","gpt-4o-mini")
    _OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _OPENAI_AVAILABLE = False

WRITING_SYSTEM = (
    "You are a strict but helpful writing tutor for grades 7–12. "
    "Follow the instruction (shorten, explain, rewrite, lengthen) while preserving meaning and citations. "
    "Prefer clarity, concision, active voice. Return ONLY the revised text."
)

def llm_rewrite(prompt: str) -> Optional[str]:
    if not _OPENAI_AVAILABLE: return None
    try:
        r = _client.chat.completions.create(
            model=_model, temperature=0.2,
            messages=[{"role":"system","content":WRITING_SYSTEM},{"role":"user","content":prompt}],
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return None

def detect_writing_task(text: str) -> Dict[str, Any]:
    low = (text or "").lower()
    if "shorten" in low or "condense" in low: 
        m = re.search(r"(?:shorten|condense).{0,40}?(\d+)\s*(?:words|w)", low)
        return {"task":"shorten","target": int(m.group(1)) if m else None}
    if any(k in low for k in ["lengthen","expand","elaborate","make it longer"]):
        m = re.search(r"(?:to|by)\s*(\d+)\s*(?:words|w|%)", low)
        return {"task":"lengthen","target": int(m.group(1)) if m else None}
    if "explain" in low or "simplify" in low: 
        m = re.search(r"(?:grade|reading level)\s*(\d+)", low)
        return {"task":"explain","grade": int(m.group(1)) if m else 9}
    if any(k in low for k in ["rewrite","rephrase","improve","fix grammar"]):
        m = re.search(r"(?:tone|style)\s*(formal|casual|academic|concise)", low)
        return {"task":"rewrite","tone": m.group(1) if m else "concise academic"}
    return {"task":"rewrite","tone":"concise academic"}

def _basic_cleanup(t: str) -> str:
    t = re.sub(r"\s+"," ", (t or "").strip())
    t = re.sub(r"\bi\b","I", t)
    parts = re.split(r"([.!?]\s+)", t)
    if len(parts) > 1:
        out=[]
        for i in range(0,len(parts),2):
            sent = parts[i].strip(); sep = parts[i+1] if i+1<len(parts) else ""
            if sent: sent = sent[:1].upper()+sent[1:]
            out.append(sent+sep)
        t="".join(out).strip()
    else:
        t = t[:1].upper()+t[1:] if t else t
    return t

def fallback_shorten(text: str, target: Optional[int]) -> str:
    words = text.split()
    if not words: return ""
    if not target: target = max(1, int(0.7*len(words)))
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out, count = [], 0
    for s in sents:
        w = len(s.split())
        if count + w <= target or not out:
            out.append(s); count += w
        else: break
    if count > target and out:
        return " ".join(" ".join(out).split()[:target])
    return " ".join(out).strip()

def fallback_lengthen(text: str, target: Optional[int]) -> str:
    base = _basic_cleanup(text)
    if target and target > 0:
        words = base.split()
        while len(words) < target:
            base += " In other words, this restates the idea with a bit more detail."
            words = base.split()
        return base
    return base + " In other words, here is a bit more detail on the same ideas."

def fallback_rewrite(text: str, tone: str) -> str:
    return _basic_cleanup(text)

def writing_tool(text: str) -> Dict[str, Any]:
    try:
        task = detect_writing_task(text)
        m = re.search(r":\s*(.+)$", text, flags=re.S)
        content = m.group(1) if m else text

        if task["task"] == "shorten":
            out = llm_rewrite(f"Shorten concisely; keep key points. Return only the result.\n\n{content}")
            return {"type":"shorten","engine":"llm" if out else "fallback","output": out or fallback_shorten(content, task.get("target"))}
        if task["task"] == "lengthen":
            out = llm_rewrite(f"Expand with more detail/examples. Return only the result.\n\n{content}")
            return {"type":"lengthen","engine":"llm" if out else "fallback","output": out or fallback_lengthen(content, task.get("target"))}
        if task["task"] == "explain":
            grade = task.get("grade",9)
            out = llm_rewrite(f"Explain at about grade {grade} level, keep essential facts. Return only the explanation.\n\n{content}")
            return {"type":"explain","engine":"llm" if out else "fallback","output": out or fallback_lengthen(_basic_cleanup(content), None)}
        tone = task.get("tone","concise academic")
        out = llm_rewrite(f"Rewrite in a {tone} tone. Improve clarity/grammar, keep meaning. Return only the result.\n\n{content}")
        return {"type":"rewrite","engine":"llm" if out else "fallback","output": out or fallback_rewrite(content, tone)}
    except Exception as e:
        return {"error": f"Writing tool failed: {e}"}

# ===================
# Code: generate OR run
# ===================
LANG_MAP = {
    "python":"python","py":"python","java":"java",
    "javascript":"javascript","js":"javascript",
    "typescript":"typescript","ts":"typescript",
    "c#":"csharp","csharp":"csharp","dotnet":"csharp",
    "c++":"cpp","cpp":"cpp","c ":"c"," go":"go","golang":"go",
    "rust":"rust","ruby":"ruby","php":"php","swift":"swift","kotlin":"kotlin",
}

def detect_language(text: str) -> str:
    low = (text or "").lower()
    for k,v in LANG_MAP.items():
        if k in low: return v
    return "python"

def codegen_hello(lang: str) -> str:
    samples = {
        "python": 'print("Hello, world!")',
        "java": 'public class Main { public static void main(String[] args){ System.out.println("Hello, world!"); } }',
        "javascript": 'console.log("Hello, world!");',
        "typescript": 'console.log("Hello, world!");',
        "c": '#include <stdio.h>\nint main(){ printf("Hello, world!\\n"); return 0; }',
        "cpp": '#include <iostream>\nint main(){ std::cout << "Hello, world!" << std::endl; return 0; }',
        "csharp": 'using System; class Program { static void Main(){ Console.WriteLine("Hello, world!"); } }',
        "go": 'package main\nimport "fmt"\nfunc main(){ fmt.Println("Hello, world!") }',
        "rust": 'fn main(){ println!("Hello, world!"); }',
        "ruby": 'puts "Hello, world!"',
        "php": '<?php echo "Hello, world!\\n";',
        "swift": 'import Foundation\nprint("Hello, world!")',
        "kotlin": 'fun main(){ println("Hello, world!") }',
    }
    return samples.get(lang, samples["python"])

def _post_json(url: str, headers: Dict[str,str], payload: Dict[str,Any], timeout: float=8.0) -> Optional[Dict[str,Any]]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception:
        return None

def code_generate_tool(text: str) -> Dict[str, Any]:
    """
    Generate code for the user's request.
    Order: OpenAI (if key) → Groq (if key) → OpenRouter (if key) → fallback.
    Always return ONLY code in 'code'.
    """
    lang = detect_language(text)

    # ---------- OpenAI ----------
    if _OPENAI_AVAILABLE:
        try:
            out = _client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Respond with ONLY code (no prose)."},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            ).choices[0].message.content.strip()
            m = re.search(r"```[^\n]*\n(.*?)```", out, flags=re.S)
            code = m.group(1).strip() if m else out
            return {"type": "generate", "engine": "llm", "language": lang, "code": code}
        except Exception:
            pass  # fall through

    # ---------- Groq ----------
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        j = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            {
                "model": "llama-3.1-70b-versatile",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Respond with ONLY code (no prose)."},
                    {"role": "user", "content": text},
                ],
            },
        )
        if j and j.get("choices"):
            out = j["choices"][0]["message"]["content"].strip()
            m = re.search(r"```[^\n]*\n(.*?)```", out, flags=re.S)
            code = m.group(1).strip() if m else out
            return {"type": "generate", "engine": "llm-groq", "language": lang, "code": code}

    # ---------- OpenRouter ----------
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        j = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"},
            {
                "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct:free"),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Respond with ONLY code (no prose)."},
                    {"role": "user", "content": text},
                ],
            },
        )
        if j and j.get("choices"):
            out = j["choices"][0]["message"]["content"].strip()
            m = re.search(r"```[^\n]*\n(.*?)```", out, flags=re.S)
            code = m.group(1).strip() if m else out
            return {"type": "generate", "engine": "llm-openrouter", "language": lang, "code": code}

    # ---------- Fallbacks ----------
    low = (text or "").lower()
    if "hello world" in low or ("print" in low and "hello" in low):
        return {"type": "generate", "engine": "fallback", "language": lang, "code": codegen_hello(lang)}

    return {
        "type": "generate",
        "engine": "fallback",
        "language": lang,
        "code": (
            "// Describe exactly what to build.\n"
            "// Example: Write a function in Python that returns the nth Fibonacci number.\n"
        ),
    }

# ----- Code runner (Python only; safe) -----
def extract_code(text: str) -> str:
    m = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.S)
    if m: return m.group(1)
    if text.lower().startswith("run:"): return text.split(":",1)[1]
    return text

def safe_run_python(code: str) -> Dict[str, Any]:
    try:
        import ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"syntax":"error","detail": f"{e.msg} at {e.lineno}:{e.offset}"}
        with tempfile.TemporaryDirectory() as td:
            main_py = os.path.join(td,"main.py")
            wrapper_py = os.path.join(td,"wrapper.py")
            open(main_py,"w",encoding="utf-8").write(code)
            wrapper = r"""
import sys, os, runpy
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
except Exception: pass
import socket, subprocess
def _block(*a, **k): raise RuntimeError("disabled")
socket.socket = _block
subprocess.Popen = _block
os.system = _block
runpy.run_path("main.py", run_name="__main__")
"""
            open(wrapper_py,"w",encoding="utf-8").write(wrapper)
            proc = subprocess.run([sys.executable,"-I","wrapper.py"], cwd=td,
                                  capture_output=True, timeout=3, text=True)
            return {"syntax":"ok","ran":True,"returncode":proc.returncode,
                    "stdout":proc.stdout[-8000:],"stderr":proc.stderr[-8000:],"timeout":False}
    except subprocess.TimeoutExpired:
        return {"syntax":"ok","ran":True,"timeout":True,"stdout":"","stderr":"Timed out after 3s"}
    except Exception as e:
        return {"error": f"Runner failed: {e}"}

def code_run_tool(text: str) -> Dict[str, Any]:
    try:
        code = extract_code(text)
        return safe_run_python(code)
    except Exception as e:
        return {"error": f"Code tool failed: {e}"}

# ===================
# General knowledge (LLM if key; else free web)
# ===================
def _get_json(url: str, timeout: float=5.0) -> Optional[Dict[str,Any]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Slate/0.6"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as r:
            return json.loads(r.read().decode("utf-8","ignore"))
    except Exception:
        return None

def _ddg_instant(q: str) -> Dict[str, Any]:
    url = "https://api.duckduckgo.com/?"+urllib.parse.urlencode({"q":q,"format":"json","no_html":"1","skip_disambig":"1"})
    j = _get_json(url) or {}
    text = (j.get("AbstractText") or "").strip()
    src  = (j.get("AbstractURL") or "").strip()
    return {"text": text, "url": src}

def _wiki_summary(q: str) -> Dict[str, Any]:
    title = urllib.parse.quote(q.strip().replace(" ","_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    j = _get_json(url) or {}
    text = (j.get("extract") or "").strip()
    page = (j.get("content_urls",{}).get("desktop",{}).get("page") or f"https://en.wikipedia.org/wiki/{title}")
    return {"text": text, "url": page}

def _llm_general(query: str) -> Optional[str]:
    # OpenAI
    if _OPENAI_AVAILABLE:
        try:
            r = _client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"system","content":"You answer AP-level questions clearly and concisely. Cite facts when needed."},
                          {"role":"user","content":query}],
                temperature=0.2,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            pass
    # Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        j = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {groq_key}","Content-Type":"application/json"},
            {"model":"llama-3.1-70b-versatile","temperature":0.2,
             "messages":[{"role":"system","content":"You answer AP-level questions clearly and concisely."},
                         {"role":"user","content":query}]}
        )
        if j and j.get("choices"):
            return j["choices"][0]["message"]["content"].strip()
    # OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        j = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"Authorization": f"Bearer {or_key}","Content-Type":"application/json"},
            {"model": os.getenv("OPENROUTER_MODEL","meta-llama/llama-3.1-70b-instruct:free"),
             "temperature":0.2,
             "messages":[{"role":"system","content":"You answer AP-level questions clearly and concisely."},
                         {"role":"user","content":query}]}
        )
        if j and j.get("choices"):
            return j["choices"][0]["message"]["content"].strip()
    return None

def general_qa_tool(query: str) -> Dict[str, Any]:
    llm = _llm_general(query)
    if llm:
        return {"type":"general","engine":"llm","answer": llm, "sources":[]}
    ddg = _ddg_instant(query)
    if ddg.get("text"):
        return {"type":"general","engine":"web","answer": ddg["text"], "sources":[ddg.get("url")] if ddg.get("url") else []}
    wiki = _wiki_summary(query)
    if wiki.get("text"):
        return {"type":"general","engine":"web","answer": wiki["text"], "sources":[wiki.get("url")] if wiki.get("url") else []}
    return {"type":"general","engine":"none","answer":"I couldn’t find a solid quick answer. Try rephrasing or ask something more specific.", "sources":[]}

# =========
# API layer
# =========
class ChatIn(BaseModel):
    mode: Literal["auto","math","write","code"] = "auto"
    text: str

def _looks_like_arithmetic(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9xXyY\.\s\+\-\*/\^\(\)%]+", s or ""))

def _has_code_fence_or_run(s: str) -> bool:
    low = (s or "").lower()
    return ("```" in low) or low.strip().startswith("run:")

def _is_code_generate_prompt(s: str) -> bool:
    low = (s or "").lower()
    verbs = ["write","generate","make","create","build","produce","print"]
    if any(v in low for v in verbs) and ("code" in low or any(k in low for k in LANG_MAP.keys())): return True
    if "hello world" in low and any(k in low for k in LANG_MAP.keys()): return True
    return False

def _is_writing_prompt(s: str) -> bool:
    low = (s or "").lower()
    return any(k in low for k in ["shorten","condense","explain","simplify","rewrite","rephrase","improve","fix grammar","lengthen","expand","elaborate"])

@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <html><body style="font-family:ui-sans-serif;background:#0b0b0c;color:#e7e7ea">
      <h3>Slate API v0.6.0</h3>
      <ul>
        <li><a href="/health" style="color:#8b5cf6">/health</a></li>
        <li><a href="/docs" style="color:#8b5cf6">/docs</a> (test POST /chat here)</li>
        <li><a href="/widget" style="color:#8b5cf6">/widget</a> · <a href="/w" style="color:#8b5cf6">/w</a> · <a href="/widget_debug" style="color:#8b5cf6">/widget_debug</a></li>
      </ul>
    </body></html>
    """

@app.get("/widget")
def widget_redirect():
    return RedirectResponse(url="/w", status_code=302)

# -------------------------------
# /w UI (with tips; same-origin calls)
# -------------------------------
NEW_WIDGET_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
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
  .hint{color:var(--muted);font-size:12px}
  .tips{background:#0f0f12;border:1px solid #222226;border-radius:10px;padding:10px;font-size:12px;color:#a1a1aa}
  .tips code{color:#e7e7ea}
</style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <strong>Slate</strong>
      <div style="flex:1"></div>
      <button class="tab" data-w-mode="auto">Auto</button>
      <button class="tab" data-w-mode="math">Math</button>
      <button class="tab" data-w-mode="code">Code</button>
      <button class="tab" data-w-mode="write">Writing</button>
    </div>

    <div class="tips">
      <div><b>How to use</b></div>
      <div>• <b>Auto</b>: just ask. Examples: <code>8*8</code>, <code>solve x^2=9</code>, <code>write hello world in java</code>, <code>shorten: ...</code>, <code>What caused the French Revolution?</code></div>
      <div>• <b>Math</b>: <i>differentiate 3x^2-4x+7</i>, <i>integral of e^x</i>, <i>roots of x^2-5x+6</i></div>
      <div>• <b>Code</b>: to <i>generate</i> code, “write/generate code in &lt;language&gt; …”. To <i>run</i> Python, paste:<br><code>```python<br>print(2+2)<br>```</code> or <code>run: print(2+2)</code></div>
      <div>• <b>Writing</b>: <code>shorten:</code>, <code>explain:</code>, <code>rewrite:</code>, <code>lengthen:</code> followed by text.</div>
      <div class="hint">Enter to send • Shift+Enter for newline</div>
    </div>

    <div id="w-chat" class="box" aria-live="polite"></div>
    <div class="input">
      <textarea id="w-input" placeholder="Try: What caused the French Revolution?  ·  8*8  ·  write hello world in java  ·  shorten: (paste text)"></textarea>
      <button id="w-send" type="button">Send</button>
    </div>
  </div>

<script>
(function(){
  const BACKEND_URL = location.origin;
  const params = new URLSearchParams(location.search);
  const defaultMode = (params.get("mode") || params.get("m") || "auto").toLowerCase();

  const chat  = document.getElementById("w-chat");
  const input = document.getElementById("w-input");
  const send  = document.getElementById("w-send");
  const tabs  = Array.from(document.querySelectorAll("[data-w-mode]"));
  let mode = "auto", sending = false, warmed = false;

  tabs.forEach(btn => {
    if (btn.dataset.wMode === defaultMode) { btn.classList.add("active"); mode = defaultMode; }
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      mode = btn.dataset.wMode;
    });
  });
  if (!tabs.some(b => b.classList.contains("active"))) {
    const autoBtn = tabs.find(b => b.dataset.wMode === "auto");
    if (autoBtn) { autoBtn.classList.add("active"); mode = "auto"; }
  }

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
      if(result.type === "simplify") return add(`Simplified ${result.expr} → ${result.simplified}`, "bot");
      if(result.type === "diff") return add(`d/dx of ${result.expr} = ${result["d/dx"]}`, "bot");
      if(result.type === "integrate") return add(`∫ ${result.expr} dx = ${result["∫dx"]}`, "bot");
      if(result.type === "solve") return add(`Equation: ${result.equation}\nSymbol: ${result.symbol}\nSolution: ${JSON.stringify(result.solution,null,2)}`, "bot");
      return add(JSON.stringify(result), "bot");
    }

    if(mode === "code"){
      if(result.error) return add("Code error: " + result.error, "bot");
      if(result.type === "generate" && result.code){ return add(result.code, "bot"); }
      if(result.syntax === "error") return add("Syntax error: " + result.detail, "bot");
      if(result.timeout) return add("Your code timed out (3s limit).", "bot");
      let out = "Ran ✅";
      if(result.stdout) out += "\\nstdout:\\n" + result.stdout.trim();
      if(result.stderr) out += "\\nstderr:\\n" + result.stderr.trim();
      return add(out, "bot");
    }

    if(mode === "write"){
      if(result.error) return add("Writing error: " + result.error, "bot");
      return add(result.output || JSON.stringify(result), "bot");
    }

    if(mode === "general"){
      if(result.error) return add("General error: " + result.error, "bot");
      let txt = result.answer || JSON.stringify(result);
      if (Array.isArray(result.sources) && result.sources.length){
        txt += "\\n\\nSources:\\n" + result.sources.join("\\n");
      }
      return add(txt, "bot");
    }

    return add(JSON.stringify(result), "bot");
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

  add("Hi! I’m Slate. Ask AP history/math/Spanish/English questions, or try 8*8, or 'write hello world in java', or 'shorten: ...'.", "bot");
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
  const BACKEND_URL = location.origin;
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
# Misc endpoints and router
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

    # math signals
    math_keywords = ["solve","root","zero","derivative","differentiate","d/dx","integral","integrate","simplify","value of","compute","evaluate","what is","what's","whats"]
    is_mathy = any(k in low for k in math_keywords) or "=" in t or "^" in t or _looks_like_arithmetic(t)

    # code signals
    code_run_like = _has_code_fence_or_run(t)
    code_generate_like = _is_code_generate_prompt(t)

    # writing signals
    writing_like = _is_writing_prompt(t)

    if mode == "math":
        return {"mode":"math","result": math_tool(t)}
    if mode == "code":
        return {"mode":"code","result": code_run_tool(t) if code_run_like else code_generate_tool(t)}
    if mode == "write":
        return {"mode":"write","result": writing_tool(t)}

    # AUTO priority: code-run > math > code-gen > writing > general
    if code_run_like:
        return {"mode":"code","result": code_run_tool(t)}
    if is_mathy:
        return {"mode":"math","result": math_tool(t)}
    if code_generate_like:
        return {"mode":"code","result": code_generate_tool(t)}
    if writing_like:
        return {"mode":"write","result": writing_tool(t)}
    # general knowledge fallback
    return {"mode":"general","result": general_qa_tool(t)}

@app.post("/chat")
def chat(payload: ChatIn):
    try:
        return route(payload.mode, payload.text)
    except Exception as e:
        return {"error": f"Server error: {e}"}
