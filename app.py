# app.py
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, List
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

app = FastAPI(title="SchoolBot API", version="0.1.0")

# --- CORS (allow all for MVP; lock down to your Wix domain in production) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    mode: Literal["auto","math","code","write"] = "auto"
    text: str
    extra: Optional[Dict[str, Any]] = None

def solve_math(text: str) -> Dict[str, Any]:
    x,y,z,t = sp.symbols('x y z t')
    try:
        low = text.lower().strip()
        # Very simple intent detection for MVP
        if "=" in text:
            left, right = text.split("=", 1)
            eq = sp.Eq(parse_expr(left), parse_expr(right))
            syms = sorted(eq.free_symbols, key=lambda s: s.name)
            sol = sp.solve(eq, syms or [x], dict=True)
            return {"type":"solve", "symbols":[str(s) for s in syms], "solution": sol}
        if low.startswith(("diff","derive","d/dx")):
            expr = parse_expr(text.split(None,1)[1])
            return {"type":"diff", "d/dx": str(sp.diff(expr, x))}
        if low.startswith(("int","integrate","∫")):
            expr = parse_expr(text.split(None,1)[1])
            return {"type":"integrate", "∫dx": str(sp.integrate(expr, x))}
        # Fallback: simplify
        expr = parse_expr(text)
        return {"type":"simplify", "simplified": str(sp.simplify(expr))}
    except Exception as e:
        return {"error": f"Math parse/solve failed: {e}"}

def analyze_code(snippet: str, language: str="python") -> Dict[str, Any]:
    if language.lower() != "python":
        return {"error":"Only Python syntax-check is implemented in MVP.", "supported":["python"]}
    import ast
    try:
        ast.parse(snippet)
        return {
            "syntax":"ok",
            "advice":[
                "Add unit tests with pytest in a sandbox (not enabled in MVP).",
                "Run ruff/flake8 and black for style & lint.",
                "Consider edge cases: empty input, None, large inputs."
            ]
        }
    except SyntaxError as e:
        return {"syntax":"error", "detail": f"{e.msg} at line {e.lineno}:{e.offset}"}

def writing_outline(prompt: str, grade: int=9, length: int=600) -> Dict[str, Any]:
    return {
        "thesis":"<Arguable thesis responding to the prompt>",
        "claims":[
            {"claim":"Distinct Claim 1", "evidence_prompts":["Key fact/example","Short quote (Author, Year)"]},
            {"claim":"Distinct Claim 2", "evidence_prompts":["Dataset/figure","Counterexample + rebuttal"]},
            {"claim":"Distinct Claim 3 (optional)", "evidence_prompts":["Historical context","Expert opinion"]}
        ],
        "structure":["Intro (hook→context→thesis)","Body 1","Body 2","Body 3","Conclusion (so what?)"],
        "revision_checklist":["Clarity","Concision","Citations present","Sentence variety","Active voice"],
        "integrity_note":"Use this as a study aid. Write in your own words and cite sources.",
        "target_grade":grade, "target_length":length
    }

def route(mode:str, text:str, extra:Optional[Dict[str,Any]]):
    text = text.strip()
    if not text:
        return {"error":"Empty prompt."}
    # crude router heuristics for MVP
    if mode=="math" or (mode=="auto" and any(k in text.lower() for k in ["solve","factor","simplify","=", "integrate","differentiate","d/dx","∫"])):
        return {"mode":"math","result":solve_math(text)}
    if mode=="code" or (mode=="auto" and any(k in text.lower() for k in ["python","java","bug","error","function","class","compile","code"])):
        lang = (extra or {}).get("language","python")
        return {"mode":"code","result":analyze_code(text, lang)}
    if mode in ("write","auto"):
        grade = int((extra or {}).get("grade", 9))
        length = int((extra or {}).get("length", 600))
        return {"mode":"write","result":writing_outline(text, grade, length)}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/chat")
def chat(payload: ChatIn):
    try:
        return route(payload.mode, payload.text, payload.extra)
    except Exception as e:
        return {"error": f"Server error: {e}"}
