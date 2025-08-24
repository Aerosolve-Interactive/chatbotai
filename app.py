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
import os, re, tempfile, subprocess, sys, json, time

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
        out=[]; 
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
        # rewrite
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
    # Try LLMs in order: OpenAI -> Groq -> OpenRouter
    lang = detect_language(text)
    # OpenAI
    if _OPENAI_AVAILABLE:
        try:
            out = _client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"system","content":"Respond with ONLY code (no prose)."},
                          {"role":"user","content":text}],
                temperature=0.2,
            ).choices[0].message.content.strip()
            m = re.search(r"```[^\n]*\n(.*?)```", out, flags=re.S)
            code = m.group(1).strip() if m else out
            return {"type":"generate","engine":"llm","language":lang,"code":code}
        except Exception:
            pass
    # Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        j = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {groq_key}","Content-Type":"application/json"},
            {"model":"llama-3.1-70b-versatile","temperature":0.2,
             "messages":[{"role":"system","content":"Respond with ONLY code (no prose)."},
                         {"role":"user","content":text}]}
        )
        if j and j.get("choices"):
            out = j["choices"][0]["message"]["content"].strip()
            m = re.search(r"```[^\n]*\n(.*?)```", out, flags=re.S)
            code = m.group(1).strip() if m else out
            return {"type":"generate","engine":"llm-groq","language":lang,"code":code}
    # OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        j = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"Authorization": f"Bearer {or_key}","Content-Type":"application/json"},
            {"model": os.getenv("OPENROUTER_MODEL","meta-llama/llama-3.1-70b-instruct:free"),
             "temperature":0.2,
             "messages":[{"role":"system","content":"Respond with ONLY code (no prose)."},
                         {"role":"user","content":text}]}
        )
        if j and j.get("choices"
