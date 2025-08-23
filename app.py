# app.py (v2: English-friendly math + index + favicon)
import re
from typing import Literal, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# ---------- FastAPI ----------
app = FastAPI(title="SchoolBot API", version="0.2.0")

# CORS: open for MVP; lock to your Wix domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Parsing helpers ----------
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # "5x" -> 5*x
    convert_xor,                          # "^"  -> "**"
)

def sp_parse(s: str):
    """SymPy parse with implicit multiplication and ^ support."""
    return parse_expr(s, transformations=TRANSFORMS)

def cleanup_math_text(t: str) -> str:
    """Normalize common english-y inputs a bit."""
    # Replace unicode minus and weird spaces
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    # Remove trailing 'for x' etc.
    t = re.sub(r"\bfor\s+x\b", "", t, flags=re.I)
    return t.strip()

def english_to_math(text: str) -> Dict[str, Any]:
    """
    Very small rule-based layer:
    - 'solve ... = ...', 'find the roots of ...', 'roots of ...'
    - 'derivative of ...', 'differentiate ...', 'd/dx ...'
    - 'integral of ...', 'antiderivative of ...', 'integrate ...'
    - 'simplify ...'
    Returns a dict spec or {} if not recognized.
    """
    raw = text.strip()
    low = raw.lower().strip()
    t = cleanup_math_text(raw)

    # Solve / roots
    if any(k in low for k in ["solve", "roots of", "root of", "zeros of", "zeroes of", "find the roots"]):
        # Prefer explicit equation
        m = re.search(r"(.+?)=\s*(.+)", t)
        if m:
            left, right = m.group(1), m.group(2)
        else:
            # try "roots of <expr>"
            m = re.search(r"(?:roots?|zeros?|zeroes?)\s+of\s+(.+)", low)
            if m:
                left, right = m.group(1), "0"
            else:
                # try "solve <expr>" -> = 0
                m = re.search(r"solve\s+(.+)", low)
                if m:
                    left, right = m.group(1), "0"
                else:
                    # not enough info
                    return {}
        return {"op":"solve", "left": left, "right": right}

    # Derivative
    if any(k in low for k in ["derivative of", "differentiate", "d/dx "]):
        expr = low
        # d/dx f(x)
        m = re.search(r"d\/dx\s+(.+)", low)
        if m:
            expr = m.group(1)
        else:
            m = re.search(r"(?:derivative of|differentiate)\s+(.+)", low)
            if m:
                expr = m.group(1)
        return {"op":"diff", "expr": expr}

    # Integral
    if any(k in low for k in ["integral of", "antiderivative of", "integrate "]):
        m = re.search(r"(?:integral of|antiderivative of|integrate)\s+(.+)", low)
        if m:
            return {"op":"integrate", "expr": m.group(1)}

    # Simplify
    if "simplify" in low:
        m = re.search(r"simplify\s+(.+)", low)
        if m:
            return {"op":"simplify", "expr": m.group(1)}

    # Fallback: check if the raw text looks like an equation or bare expression
    if "=" in t:
        left, right = t.split("=", 1)
        return {"op":"solve", "left": left, "right": right}
    # bare expression -> try simplify
    return {"op":"simplify", "expr": t}

# ---------- Tools ----------
def math_tool(text: str) -> Dict[str, Any]:
    spec = english_to_math(text)
    try:
        if not spec:
            return {"error":"Unrecognized math intent. Try: 'solve x^2-5x+6=0', 'derivative of ...', 'integral of ...', or 'simplify ...'."}
        if spec["op"] == "solve":
            left, right = sp_parse(spec["left"]), sp_parse(spec["right"])
            eq = sp.Eq(left, right)
            syms = sorted(eq.free_symbols, key=lambda s: s.name) or [sp.symbols("x") ]
            sol = sp.solve(eq, syms[0], dict=True)
            return {"type":"solve", "equation": str(eq), "symbol": str(syms[0]), "solution": sol}
        if spec["op"] == "diff":
            x = sp.symbols("x")
            expr = sp_parse(spec["expr"])
            return {"type":"diff", "expr": str(expr), "d/dx": str(sp.diff(expr, x))}
        if spec["op"] == "integrate":
            x = sp.symbols("x")
            expr = sp_parse(spec["expr"])
            return {"type":"integrate", "expr": str(expr), "∫dx": str(sp.integrate(expr, x))}
        if spec["op"] == "simplify":
            expr = sp_parse(spec["expr"])
            return {"type":"simplify", "expr": str(expr), "simplified": str(sp.simplify(expr))}
        return {"error":"Unknown math op."}
    except Exception as e:
        return {"error": f"Math parse/solve failed: {e}"}

def code_tool(snippet: str, language: str="python") -> Dict[str, Any]:
    if language.lower() != "python":
        return {"error":"Only Python syntax-check MVP. Add Docker sandbox for execution."}
    import ast
    try:
        ast.parse(snippet)
        return {"syntax":"ok","advice":["Add pytest tests","Run ruff/black","Handle edge cases"]}
    except SyntaxError as e:
        return {"syntax":"error","detail": f"{e.msg} at {e.lineno}:{e.offset}"}

def writing_tool(prompt: str, grade: int=9, length: int=600) -> Dict[str, Any]:
    return {
        "thesis":"<Arguable thesis responding to the prompt>",
        "claims":[
            {"claim":"Distinct Claim 1", "evidence_prompts":["Key fact/example","Short quote (Author, Year)"]},
            {"claim":"Distinct Claim 2", "evidence_prompts":["Dataset/figure","Counterexample + rebuttal"]},
            {"claim":"Distinct Claim 3 (optional)", "evidence_prompts":["Historical context","Expert opinion"]}
        ],
        "structure":["Intro (hook→context→thesis)","Body 1","Body 2","Body 3","Conclusion (so what?)"],
        "revision_checklist":["Clarity","Concision","Citations present","Sentence variety","Active voice"],
        "integrity_note":"Use as a study aid. Write in your own words and cite sources.",
        "target_grade":grade, "target_length":length
    }

# ---------- API ----------
class ChatIn(BaseModel):
    mode: Literal["auto","math","code","write"] = "auto"
    text: str
    extra: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><body style="font-family:ui-sans-serif;background:#0b0b0c;color:#e7e7ea">
      <h3>SchoolBot API v0.2</h3>
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
    return {"status":"ok"}

def route(mode: str, text: str, extra: Optional[Dict[str, Any]]):
    t = (text or "").strip()
    if not t:
        return {"error":"Empty prompt."}
    low = t.lower()
    if mode=="math" or (mode=="auto" and any(k in low for k in ["solve","root","zero","derivative","differentiate","d/dx","integral","integrate","simplify","^","="])):
        return {"mode":"math", "result": math_tool(t)}
    if mode=="code" or (mode=="auto" and any(k in low for k in ["python","java","bug","error","function","class","compile","code"])):
        lang = (extra or {}).get("language","python")
        return {"mode":"code","result": code_tool(t, lang)}
    # default -> writing coach
    grade = int((extra or {}).get("grade", 9))
    length = int((extra or {}).get("length", 600))
    return {"mode":"write","result": writing_tool(t, grade, length)}

@app.post("/chat")
def chat(payload: ChatIn):
    try:
        return route(payload.mode, payload.text, payload.extra)
    except Exception as e:
        return {"error": f"Server error: {e}"}
