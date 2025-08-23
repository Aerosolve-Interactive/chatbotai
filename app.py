# app.py — Slate API v0.4 (English math + writing + safe code runner; no 500s)
from typing import Literal, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import os, re, tempfile, subprocess, sys, textwrap, json

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

app = FastAPI(title="Slate API", version="0.4")

# --- CORS: open for MVP; lock to your Wix domain later ---
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
    implicit_multiplication_application,  # "5x" -> 5*x
    convert_xor,                          # "^"  -> "**"
)

def sp_parse(s: str):
    s = (s or "").strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("ln", "log")
    return parse_expr(s, transformations=TRANSFORMS)

def english_to_math(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    low = raw.lower()

    # Solve explicit equation
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
        L, R = (eq.split("=", 1) + ["0"])[:2] if "=" in eq else (eq, "0")
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

    # Evaluate
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
            sol = sp.solve(eq, var, dict=True)
            return {"type": "solve", "equation": str(eq), "symbol": str(var), "solution": sol}
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
        # fallback
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
        # quick syntax check
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
                "stdout": proc.stdout[-8000:],  # cap
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
      <h3>Slate API v0.4</h3>
      <ul>
        <li><a href="/health" style="color:#8b5cf6">/health</a></li>
        <li><a href="/docs" style="color:#8b5cf6">/docs</a> (test POST /chat here)</li>
      </ul>
    </body></html>
    """

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
