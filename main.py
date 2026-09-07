"""
Pocket Me — AI Brain backend (FastAPI + Google Gemini)
=======================================================

Run it:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key-from-aistudio.google.com"
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    On Windows PowerShell:
        $env:GEMINI_API_KEY = "your-key"
        python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    Use "python -m uvicorn", NOT a bare "uvicorn": pip installs uvicorn.exe into
    a Scripts folder that is usually not on PATH, so the bare command fails with
    "'uvicorn' is not recognized". Or just run ./start.ps1, which handles both.

Model selection:
    Google retires Gemini models on a rolling basis, so no model name is
    hardcoded. On first use the service asks the API which models your key can
    actually call and picks the newest stable Flash model. To pin one:
        export GEMINI_MODEL="gemini-flash-latest"
    See what your key can call:  GET /api/v1/brain/models

Pair it with the frontend:
    In index.html set:
        const BRAIN_API_URL = 'http://127.0.0.1:8000/api/v1/brain';
    If the backend runs on your computer and the app runs in Spck on your
    phone, use the computer's LAN IP instead (same Wi-Fi), e.g.
        const BRAIN_API_URL = 'http://192.168.1.23:8000/api/v1/brain';
    Keep it http (not https) while testing locally.

Security:
    The frontend sends the signed-in user's Supabase JWT as
    "Authorization: Bearer <token>". By default this server runs in DEV
    MODE and accepts any bearer token. To verify tokens properly, set:
        export SUPABASE_JWT_SECRET="<Supabase → Settings → API → JWT Secret>"
    (Projects on Supabase's newer asymmetric signing keys should verify
    via the project JWKS instead — swap the decode call accordingly.)

Design rule:
    The app's deterministic engine remains the source of truth for every
    figure. This service only SUGGESTS category keys (validated against
    the app's exact key list on both sides) and answers questions in
    plain text. It never computes or overrides statement numbers.
"""

import json
import os
import re
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

# Optional — only needed when SUPABASE_JWT_SECRET is set.
try:
    import jwt as pyjwt
except ImportError:  # pragma: no cover
    pyjwt = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Model selection.
#   Google retires Gemini models on a rolling basis ("no longer available to new
#   users"), so pinning a literal name here guarantees a 404 sooner or later.
#   Instead we ask the API which models THIS key can actually call and pick the
#   best one. Set GEMINI_MODEL to pin an exact name and skip discovery.
GEMINI_MODEL_OVERRIDE = os.environ.get("GEMINI_MODEL", "").strip()
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "").strip()

try:
    # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
    client = genai.Client()
    _client_error = None
except Exception as exc:  # noqa: BLE001
    client = None
    _client_error = str(exc)

# Model families this service must never pick: they can't do chat completions.
_MODEL_EXCLUDE = (
    "embedding", "aqa", "imagen", "veo", "tts", "image-generation",
    "learnlm", "gemma", "robotics", "computer-use",
)
_resolved_model: Optional[str] = None


def _model_score(name: str) -> float:
    """Rank candidate models: newest stable Flash first (fast + cheap for this workload)."""
    n = name.lower().removeprefix("models/")
    score = 0.0

    # "-latest" aliases track Google's current release and never go stale — prefer them.
    if n.endswith("-latest"):
        score += 1000

    # Version number: gemini-3-pro -> 300, gemini-2.5-flash -> 250.
    m = re.search(r"gemini-(\d+)(?:[.-](\d+))?", n)
    if m:
        score += int(m.group(1)) * 100 + int(m.group(2) or 0) * 10

    if "flash" in n:
        score += 50          # matches this service's cost/latency profile
    elif "pro" in n:
        score += 30
    if "lite" in n:
        score -= 15          # cheapest tier, weaker at structured output
    if any(t in n for t in ("preview", "-exp", "experimental")):
        score -= 200         # never auto-select a preview build
    return score


def list_available_models() -> List[str]:
    """Every model this API key may call for generateContent, best candidate first."""
    if client is None:
        return []
    names: List[str] = []
    for m in client.models.list():
        name = (getattr(m, "name", "") or "").removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        # Some endpoints omit supported_actions; keep those and let scoring decide.
        if actions and "generateContent" not in actions:
            continue
        if not name or any(bad in name.lower() for bad in _MODEL_EXCLUDE):
            continue
        names.append(name)
    return sorted(names, key=_model_score, reverse=True)


def resolve_model(force: bool = False) -> str:
    """Pick a usable model, caching the result. GEMINI_MODEL always wins if set."""
    global _resolved_model
    if GEMINI_MODEL_OVERRIDE:
        return GEMINI_MODEL_OVERRIDE
    if _resolved_model and not force:
        return _resolved_model
    try:
        candidates = list_available_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Could not list Gemini models for your API key: {exc}",
        ) from exc
    if not candidates:
        raise HTTPException(
            status_code=502,
            detail="Your Gemini API key cannot call any text-generation models. "
                   "Check the key at aistudio.google.com, or set GEMINI_MODEL explicitly.",
        )
    _resolved_model = candidates[0]
    print(f"[gemini] using model: {_resolved_model}")
    return _resolved_model


def _is_model_unavailable(exc: Exception) -> bool:
    """True for 404 / NOT_FOUND / retired-model errors, which a re-resolve can fix."""
    text = str(exc).lower()
    return "not_found" in text or "404" in text or "no longer available" in text

# The app's exact category keys (from the CATEGORIES engine in index.html).
# Suggestions are validated against this list on the server AND again in the
# frontend, so an invalid key can never reach the statements.
VALID_CATEGORY_KEYS = [
    "revenue", "other_income", "cos", "opex", "finance_costs", "tax",
    "ppe", "intangibles", "investments_nc", "inventory", "receivables",
    "prepayments", "investments_c", "cash", "other_nca", "other_ca",
    "share_capital", "retained_earnings", "reserves", "drawings",
    "other_equity", "loans_nc", "deferred_tax", "other_ncl", "payables",
    "accruals", "tax_payable", "loans_c", "overdraft", "other_cl",
]
KEY_GLOSSARY = """
revenue            Sales / trading income
other_income       Interest received, sundry income
cos                Cost of sales / purchases / direct costs
opex               Operating expenses (rent, salaries, advertising, etc.)
finance_costs      Interest paid, bank finance charges
tax                Income tax EXPENSE for the period
ppe                Property, plant & equipment / fixed assets
intangibles        Goodwill, software, licences
investments_nc     Long-term investments
other_nca          Other non-current assets
inventory          Stock / inventory on hand
receivables        Trade & other receivables / accounts receivable / debtors
prepayments        Prepaid expenses
investments_c      Short-term investments
cash               Bank accounts, petty cash, cash on hand
other_ca           Other current assets
share_capital      Share capital / members' contribution
retained_earnings  Retained earnings / accumulated profit (opening)
reserves           Revaluation or other reserves
drawings           Drawings / dividends paid
other_equity       Other equity items
loans_nc           Long-term loans / bonds payable
deferred_tax       Deferred tax liability
other_ncl          Other non-current liabilities
payables           Trade & other payables / accounts payable / creditors
accruals           Accrued expenses
tax_payable        Current tax payable / VAT control / SARS payable
loans_c            Short-term loans / current portion of long-term debt
overdraft          Bank overdraft
other_cl           Other current liabilities
IGNORE             Not a real account (heading, total, control row)
"""

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Pocket Me — AI Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your deployed frontend origin later
    allow_credentials=False,    # we use an Authorization header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth — Supabase JWT (dev mode unless SUPABASE_JWT_SECRET is set)
# ---------------------------------------------------------------------------
async def require_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token — sign in again.")
    token = authorization.split(" ", 1)[1].strip()
    if not SUPABASE_JWT_SECRET:
        return {"sub": "dev-unverified"}  # DEV MODE: token accepted without verification
    if pyjwt is None:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET is set but PyJWT is not installed — pip install pyjwt")
    try:
        return pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Invalid or expired session token: {exc}") from exc


# ---------------------------------------------------------------------------
# Request / response models (match the frontend contract exactly)
# ---------------------------------------------------------------------------
class TBRow(BaseModel):
    account: str
    accountType: Optional[str] = None
    debit: float = 0
    credit: float = 0


class MapTBRequest(BaseModel):
    rows: List[TBRow]


class CategorizeRequest(BaseModel):
    description: str
    amount: Optional[float] = None
    date: Optional[str] = None


class RouterRequest(BaseModel):
    question: str
    context: Optional[dict] = None


# Structured-output shapes for Gemini
class AccountMapping(BaseModel):
    account: str
    category: str


class CategorizeResult(BaseModel):
    category: str
    confidence: float
    rationale: str


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def _require_client() -> None:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini client not initialised — set GEMINI_API_KEY and restart. ({_client_error})",
        )


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.MULTILINE).strip()


def _call_gemini(user: str, config: types.GenerateContentConfig):
    """One request, with a single retry on a newly-resolved model if ours was retired."""
    _require_client()
    model = resolve_model()
    try:
        return client.models.generate_content(model=model, contents=user, config=config)
    except Exception as exc:  # noqa: BLE001
        if _is_model_unavailable(exc):
            # The pinned/cached model disappeared — re-discover and try once more.
            try:
                retry_model = resolve_model(force=True) if not GEMINI_MODEL_OVERRIDE else model
            except HTTPException:
                retry_model = model
            if retry_model != model:
                try:
                    return client.models.generate_content(model=retry_model, contents=user, config=config)
                except Exception as exc2:  # noqa: BLE001
                    raise _gemini_error(exc2, retry_model) from exc2
            raise _gemini_error(exc, model) from exc
        raise _gemini_error(exc, model) from exc


def _gemini_error(exc: Exception, model: str) -> HTTPException:
    if _is_model_unavailable(exc):
        hint = (
            f"The model '{model}' is not available to your API key. "
            "GET /api/v1/brain/models to see what your key can call, then set "
            "GEMINI_MODEL to one of those names and restart."
        )
        return HTTPException(status_code=502, detail=hint)
    return HTTPException(status_code=502, detail=f"Gemini request failed: {exc}")


def gemini_structured(system: str, user: str, schema, temperature: float):
    """Call Gemini with a JSON response schema; fall back to manual parsing."""
    resp = _call_gemini(
        user,
        types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed
    try:
        return json.loads(_strip_fences(resp.text))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Gemini returned a response that could not be parsed as JSON.") from exc


def gemini_text(system: str, user: str, temperature: float) -> str:
    resp = _call_gemini(
        user,
        types.GenerateContentConfig(system_instruction=system, temperature=temperature),
    )
    text = (resp.text or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Gemini returned an empty answer — try again.")
    return text


def _field(item, name: str):
    """Read a field from either a Pydantic object or a plain dict."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
MAPPER_SYSTEM = f"""You are the account-classification assistant inside "Pocket Me",
a financial statements app used by a South African trainee accountant.
You receive trial-balance rows the deterministic engine could not classify
from the account name alone. Map each account to EXACTLY ONE category key.

Allowed category keys (use these strings verbatim, nothing else):
{KEY_GLOSSARY}

Rules:
- Only include accounts you are confident about. If genuinely unsure, OMIT
  the account entirely — a human reviews everything you suggest.
- Use IGNORE only for rows that are clearly not accounts (headings, totals).
- Use the debit/credit magnitudes and any accountType hint to decide the
  natural side (e.g. a credit balance named "Members Loan" is a liability).
- South African terminology: Debtors = receivables, Creditors = payables,
  VAT/SARS control = tax_payable, Members contribution = share_capital.
Return JSON only.
"""

CATEGORIZER_SYSTEM = f"""You classify a single business transaction into one category key
for the "Pocket Me" accounting app. Allowed keys (verbatim):
{KEY_GLOSSARY}

Return JSON with: category (one key), confidence (0 to 1), and a rationale
of at most 20 words. If nothing fits, use the closest key with low confidence.
"""

ROUTER_SYSTEM = """You are the financial consultant built into "Pocket Me". You are a
seasoned Chartered Accountant (CA(SA)) and CFO-level advisor: deeply IFRS-aware,
numerate, and practical about running a real business.

You are advising THIS specific user about THEIR OWN business — not a textbook
case. The context you receive has two parts: a "profile" object describing who
they are and what they care about, and a set of live financial metrics computed
by the app's deterministic engine (currency symbol included). Treat yourself as
their personal accountant who already knows them.

Use the profile to personalise everything:
- Speak to their situation directly and tailor advice to their stated goals
  and risk appetite (e.g. a cautious owner protecting cash vs. an aggressive
  one chasing growth get different emphasis).
- Match your tone to the reader. If they are a business owner rather than an
  accountant, explain in plain business language and skip the jargon; if they
  are a finance professional, you may be more technical.
- Honour their reporting framework (e.g. IFRS for SMEs vs full IFRS) and their
  tax regime (e.g. South African SARS / VAT) when it is relevant.
- Take anything in their free-text notes into account.
- If the profile is empty, give sound general guidance and, where useful,
  invite them to complete their business profile so you can be more specific.

Hard rules:
- Base every number you mention ONLY on the metrics provided. If a figure is
  null or absent, say you don't have it — never invent, estimate, or extrapolate
  figures.
- Be direct, warm, and concise: at most ~180 words.
- PLAIN TEXT ONLY — no markdown, no asterisks, no headings (the app renders your
  answer as raw text).
- This is guidance to support the owner's judgement, not a formal audit opinion,
  and not a personal investment recommendation.
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def health():
    model, model_error = None, None
    if client is not None:
        try:
            model = resolve_model()
        except HTTPException as exc:
            model_error = exc.detail
    return {
        "ok": True,
        "service": "pocket-me-brain",
        "model": model,
        "model_source": "GEMINI_MODEL env var" if GEMINI_MODEL_OVERRIDE else "auto-discovered",
        "model_error": model_error,
        "gemini_ready": client is not None and model is not None,
        "auth_mode": "verified" if SUPABASE_JWT_SECRET else "DEV (unverified tokens accepted)",
    }


@app.get("/api/v1/brain/models")
async def list_models():
    """Diagnostic: exactly which models this API key can call, best candidate first."""
    _require_client()
    try:
        names = list_available_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not list models: {exc}") from exc
    return {"selected": resolve_model() if names else None, "available": names, "count": len(names)}


@app.post("/api/v1/brain/map-trial-balance")
async def map_trial_balance(body: MapTBRequest, user: dict = Depends(require_user)):
    if not body.rows:
        return {"classifications": {}}

    rows_text = "\n".join(
        f"- account: {r.account!r} | type hint: {r.accountType or 'none'} | debit: {r.debit} | credit: {r.credit}"
        for r in body.rows
    )
    user_prompt = (
        "Classify these unmapped trial-balance rows. Return a JSON array of "
        "{account, category} objects, using each account name exactly as given:\n"
        + rows_text
    )
    items = gemini_structured(MAPPER_SYSTEM, user_prompt, list[AccountMapping], temperature=0.1)

    sent_accounts = {r.account for r in body.rows}
    classifications = {}
    for item in items or []:
        account = str(_field(item, "account") or "").strip()
        key = str(_field(item, "category") or "").strip()
        if account in sent_accounts and (key == "IGNORE" or key in VALID_CATEGORY_KEYS):
            classifications[account] = key

    print(f"[map-trial-balance] user={user.get('sub')} sent={len(body.rows)} mapped={len(classifications)}")
    return {"classifications": classifications}


@app.post("/api/v1/brain/categorize-transaction")
async def categorize_transaction(body: CategorizeRequest, user: dict = Depends(require_user)):
    user_prompt = (
        "Classify this single transaction:\n"
        f"- description: {body.description!r}\n"
        f"- amount: {body.amount if body.amount is not None else 'unknown'}\n"
        f"- date: {body.date or 'unknown'}"
    )
    result = gemini_structured(CATEGORIZER_SYSTEM, user_prompt, CategorizeResult, temperature=0.1)

    key = str(_field(result, "category") or "").strip()
    try:
        confidence = max(0.0, min(1.0, float(_field(result, "confidence") or 0)))
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = str(_field(result, "rationale") or "").strip()

    valid = key == "IGNORE" or key in VALID_CATEGORY_KEYS
    print(f"[categorize-transaction] user={user.get('sub')} -> {key if valid else 'INVALID:' + key}")
    return {"category": key if valid else None, "confidence": confidence, "rationale": rationale}


@app.post("/api/v1/brain/reasoning-router")
async def reasoning_router(body: RouterRequest, user: dict = Depends(require_user)):
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is empty.")

    context = body.context if isinstance(body.context, dict) else {}
    profile = context.get("profile")
    metrics = {k: v for k, v in context.items() if k != "profile"}

    profile_json = (
        json.dumps(profile, indent=2, default=str)
        if profile
        else "No business profile was provided — give sound general guidance and, where it helps, invite them to complete their profile."
    )
    metrics_json = (
        json.dumps(metrics, indent=2, default=str)
        if metrics
        else "No live figures were provided."
    )
    user_prompt = (
        f"USER PROFILE (the person and business you are advising):\n{profile_json}\n\n"
        f"LIVE FINANCIAL METRICS (the only figures you may cite):\n{metrics_json}\n\n"
        f"THEIR QUESTION:\n{question}"
    )
    answer = gemini_text(ROUTER_SYSTEM, user_prompt, temperature=0.4)

    has_profile = bool(profile)
    print(f"[reasoning-router] user={user.get('sub')} profile={has_profile} q={question[:60]!r}")
    return {"answer": answer}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    if not SUPABASE_JWT_SECRET:
        print("WARNING: DEV MODE — bearer tokens are accepted without verification. "
              "Set SUPABASE_JWT_SECRET to enforce Supabase JWT checks.")
    # --host 0.0.0.0 lets your phone reach this server over the LAN.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)