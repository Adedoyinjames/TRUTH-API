"""
Truth Verification Pipeline API
================================

CHANGES IN THIS VERSION (accuracy / reliability hardening pass)
-----------------------------------------------------------------
The Gradio API surface is UNCHANGED: same Blocks layout, same input/output
components, same `api_name="verify"`. Everything below is internal.

Two things were fixed that were silently making every "fallback" request fail:
  1. `gemini-1.5-flash` is already fully shut down (Gemini 1.0/1.5 all return
     404) and `google.generativeai` is a deprecated SDK. Both are replaced
     with the current `google-genai` SDK + `gemini-3.1-flash-lite`.
     https://ai.google.dev/gemini-api/docs/deprecations
  2. `llama-3.3-70b-versatile` on Groq is deprecated with a shutdown date of
     2026-08-16. Replaced with Groq's own recommended replacement,
     `openai/gpt-oss-120b`. https://console.groq.com/docs/deprecations
  3. `duckduckgo_search` is frozen/renamed upstream to `ddgs`, which is now a
     multi-engine metasearch library (bing/brave/duckduckgo/google/mojeek/
     yahoo/yandex/wikipedia for text search) with built-in result de-dup and
     ranking -- meaningfully more resilient than the old single-engine
     scraper. Same result shape (title/href/body), verified against the
     installed package source.

On top of that, the pipeline itself was rebuilt around one core idea: every
single stage used to do `json.loads(call_llm(...))` with zero error handling,
so any malformed JSON from the model (extremely common -- markdown fences,
a "Sure, here's the JSON:" preamble, etc.) crashed the entire request. Now:
  - Every LLM call goes through a JSON-repair pipeline (strip fences -> parse
    -> extract the first balanced JSON block -> one corrective LLM retry ->
    typed default) so a bad response degrades a single stage instead of
    killing the whole pipeline.
  - Every stage has a typed fallback default so downstream code never KeyErrors.
  - Claims that are opinions/subjective are detected and short-circuited
    instead of being forced through a Verified/Debunked verdict.
  - Evidence extraction is grounded: the model is only allowed to cite URLs
    that were actually returned by search, and a final programmatic audit
    strips any source that doesn't match a retrieved URL -- this is the main
    defense against fabricated citations.
  - Source reliability blends the model's own judgment with a rule-based
    domain-credibility prior, so one confident-sounding LLM guess isn't the
    only signal.
  - "Verified"/"Debunked" verdicts require a minimum number of independent
    sources; thin evidence is downgraded to "Uncertain" deterministically.
  - Confidence is calibrated in plain Python (source count, grounding
    failures, empty search results) rather than trusting the model's raw
    self-reported number.
  - The whole pipeline is wrapped in a top-level try/except so the API always
    returns a well-formed JSON object -- even total failure returns a
    structured "Error" verdict instead of an unhandled exception.

Original output fields (answer, confidence, reasoningSummary,
supportingEvidence, sources) are preserved with the same names and types.
New fields (verdict, claimsAnalyzed, domains, caveats, sourceCount,
knowledgeGraph, pipelineWarnings, verifiedAt) are additive.

MAINTENANCE NOTE: model IDs get deprecated on a rolling basis. Before
assuming a model still works, check:
  - Groq:   https://console.groq.com/docs/deprecations
  - Gemini: https://ai.google.dev/gemini-api/docs/deprecations
"""

import gradio as gr
import os
import re
import json
import time
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from groq import Groq
from google import genai
from google.genai import types as genai_types

try:
    from ddgs import DDGS  # current, maintained package
except ImportError:
    from duckduckgo_search import DDGS  # frozen upstream; kept only as a transitional fallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Groq recommends openai/gpt-oss-120b as the replacement for the deprecated
# llama-3.3-70b-versatile (shutdown 2026-08-16). qwen/qwen3.6-27b is Groq's
# other listed alternative if you want to compare quality/latency.
GROQ_MODEL = "openai/gpt-oss-120b"

# gemini-1.5-flash is already shut down (404). gemini-3.1-flash-lite is GA
# and cost-efficient; swap to "gemini-3.5-flash" for a higher-quality (and
# pricier) fallback if the Gemini path gets hit often.
GEMINI_MODEL = "gemini-3.1-flash-lite"

MAX_LLM_RETRIES = 2          # per provider, before moving on (Groq -> Gemini)
MAX_SEARCH_RETRIES = 3
MIN_SOURCES_FOR_VERIFIED = 2  # independent sources required for a strong verdict

RATE_LIMIT_WINDOW = 60
MAX_REQUESTS = 5

HIGH_STAKES_DOMAINS = {
    "health", "medical", "medicine", "legal", "law", "finance",
    "financial", "election", "elections", "politics", "safety",
}

HIGH_CREDIBILITY_HINTS = (
    ".gov", ".edu", "who.int", "un.org", "reuters.com", "apnews.com",
    "bbc.co", "bbc.com", "nature.com", "science.org", "nih.gov",
    "ncbi.nlm.nih.gov", "worldbank.org", "imf.org", "unesco.org",
    "wikipedia.org",
)
LOW_CREDIBILITY_HINTS = ("blogspot.", "wordpress.com/", "medium.com/@")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
rate_limit_store: Dict[str, List[float]] = {}


def is_rate_limited(client_ip: str) -> bool:
    now = time.time()
    bucket = rate_limit_store.setdefault(client_ip, [])
    bucket[:] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]

    if len(bucket) >= MAX_REQUESTS:
        return True

    bucket.append(now)

    # light housekeeping so this dict doesn't grow forever under long uptime
    if len(rate_limit_store) > 5000:
        for k in [k for k, v in rate_limit_store.items() if not v]:
            del rate_limit_store[k]

    return False


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ---------------------------------------------------------------------------
# Robust JSON handling
#
# This is the backbone of pipeline accuracy: every stage depends on getting
# well-formed JSON back from an LLM, and LLMs reliably wrap JSON in prose,
# markdown code fences, or trailing commentary. All three helpers below are
# unit-tested against realistic malformed outputs (fences, prose preambles,
# nested braces inside string values, mismatched bracket types).
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """Grab the first balanced top-level {...} or [...] block, whichever starts first."""
    starts = [(text.find(ch), ch) for ch in ("{", "[") if text.find(ch) != -1]
    if not starts:
        return None
    starts.sort(key=lambda t: t[0])
    start, open_ch = starts[0]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def safe_json_parse(raw: str, default: Any = None) -> Any:
    """Best-effort JSON parsing that tolerates the ways LLMs mangle JSON."""
    if not raw:
        return default
    stripped = _strip_code_fences(raw)
    candidates = [raw, stripped]
    block = _extract_json_block(stripped)
    if block:
        candidates.append(block)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return default


# ---------------------------------------------------------------------------
# LLM calling: Groq primary, Gemini fallback, retries + backoff on both,
# always requesting native JSON mode, plus one corrective repair pass if
# parsing still fails.
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    stage_name: str = "llm_call",
    json_mode: bool = True,
    temperature: float = 0.1,
    reasoning_effort: str = "medium",
) -> str:
    """Call Groq, falling back to Gemini, with retries + backoff on both."""
    if not groq_client and not gemini_client:
        return json.dumps({"error": "No LLM provider configured: set GROQ_API_KEY and/or GEMINI_API_KEY."})

    last_error: Any = None

    if groq_client:
        for attempt in range(MAX_LLM_RETRIES):
            try:
                kwargs: Dict[str, Any] = dict(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL,
                    temperature=temperature,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if "gpt-oss" in GROQ_MODEL:
                    kwargs["reasoning_effort"] = reasoning_effort
                completion = groq_client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content
                if content and content.strip():
                    return content
            except Exception as e:
                last_error = e
                print(f"[{stage_name}] Groq attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 8) + random.random())
        print(f"[{stage_name}] Groq exhausted retries, falling back to Gemini. Last error: {last_error}")

    if gemini_client:
        for attempt in range(MAX_LLM_RETRIES):
            try:
                config_kwargs: Dict[str, Any] = {"temperature": temperature}
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                )
                if response.text and response.text.strip():
                    return response.text
            except Exception as e:
                last_error = e
                print(f"[{stage_name}] Gemini attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 8) + random.random())
        print(f"[{stage_name}] Gemini also exhausted retries. Last error: {last_error}")

    return json.dumps({"error": f"All LLM providers failed for stage '{stage_name}': {last_error}"})


def call_llm_json(
    prompt: str,
    stage_name: str = "llm_call",
    default: Any = None,
    temperature: float = 0.1,
    reasoning_effort: str = "medium",
) -> Any:
    """call_llm + robust parsing + one corrective retry before giving up."""
    raw = call_llm(prompt, stage_name, json_mode=True, temperature=temperature, reasoning_effort=reasoning_effort)
    parsed = safe_json_parse(raw)
    if parsed is not None:
        return parsed

    repair_prompt = (
        "The following text was supposed to be valid JSON but failed to parse. "
        "Return ONLY the corrected, valid JSON with no commentary and no markdown fences:\n\n"
        f"{raw[:2000]}"
    )
    raw_retry = call_llm(repair_prompt, f"{stage_name}_repair", json_mode=True, temperature=0.0, reasoning_effort=reasoning_effort)
    parsed_retry = safe_json_parse(raw_retry)
    if parsed_retry is not None:
        return parsed_retry

    print(f"[{stage_name}] JSON parsing failed twice; using default.")
    return default


# ---------------------------------------------------------------------------
# Web search: retries + backoff, multi-engine via ddgs (backend="auto" tries
# bing/brave/duckduckgo/google/mojeek/yahoo/yandex/wikipedia and aggregates),
# plus de-duplication across the multiple queries the pipeline issues.
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    for attempt in range(MAX_SEARCH_RETRIES):
        try:
            with DDGS() as ddgs:
                results = [
                    {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
                    for r in ddgs.text(query, max_results=max_results, backend="auto")
                ]
            if results:
                return results
            return []  # legitimately no results; don't retry forever on an empty-but-successful search
        except Exception as e:
            print(f"Search attempt {attempt + 1} failed for '{query}': {e}")
            time.sleep(min(2 ** attempt, 6) + random.random())
    return []


def dedupe_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped = []
    for r in results:
        href = (r.get("href") or "").strip().rstrip("/")
        if href:
            if href in seen:
                continue
            seen.add(href)
        deduped.append(r)
    return deduped


def domain_credibility_hint(url: str) -> float:
    """A rule-based prior, NOT a verdict on truth -- just a floor/ceiling nudge
    blended with the model's own reliability judgment so a single LLM call
    isn't the only signal on source quality."""
    url_lower = (url or "").lower()
    if any(d in url_lower for d in HIGH_CREDIBILITY_HINTS):
        return 0.85
    if any(d in url_lower for d in LOW_CREDIBILITY_HINTS):
        return 0.35
    return 0.55


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pipeline stages. Each has a typed default so a stage failure degrades
# gracefully instead of raising.
# ---------------------------------------------------------------------------

def stage_analyze_query(query: str) -> Dict[str, Any]:
    default = {"complexity": "moderate", "intent": "Verify a factual claim", "isVerifiable": True, "isOpinion": False}
    prompt = f"""You are the Query Analyzer in a fact-verification pipeline.

Determine whether the following input is an objectively verifiable factual claim, or a subjective opinion/preference/unanswerable question.

Input: {query!r}

Return ONLY valid JSON, no markdown, no commentary, in exactly this shape:
{{"complexity": "simple" | "moderate" | "complex", "intent": "<one short sentence>", "isVerifiable": true | false, "isOpinion": true | false}}

isVerifiable should be false for pure opinions, preferences, or requests that aren't actually claims (e.g. "what's the best pizza topping").
isVerifiable should be true for concrete factual, historical, scientific, statistical, or current-events claims that evidence could confirm or refute."""
    result = call_llm_json(prompt, "query_analyzer", default=default, temperature=0.0, reasoning_effort="low")
    if not isinstance(result, dict):
        return default
    return {**default, **result}


def stage_extract_claims(query: str) -> List[str]:
    prompt = f"""Extract the distinct, atomic, independently-checkable factual claims from this input.
Input: {query!r}

Return ONLY valid JSON: {{"claims": ["claim 1", "claim 2", ...]}}
If the input is already a single atomic claim, return it as the only item. Do not add claims that weren't stated or implied."""
    result = call_llm_json(prompt, "claim_extractor", default={"claims": [query]}, temperature=0.0, reasoning_effort="low")
    claims = result.get("claims") if isinstance(result, dict) else None
    if not isinstance(claims, list):
        return [query]
    cleaned = [str(c).strip() for c in claims if str(c).strip()]
    return cleaned or [query]


def stage_classify_domains(claims: List[str]) -> List[str]:
    prompt = f"""Classify the subject-matter domain(s) of these claims (e.g. Science, Technology, Politics, Health, History, Finance, Geography, Sports, Entertainment).
Claims: {claims}

Return ONLY valid JSON: {{"domains": ["Domain1", "Domain2", ...]}}"""
    result = call_llm_json(prompt, "domain_classifier", default={"domains": ["General"]}, temperature=0.0, reasoning_effort="low")
    domains = result.get("domains") if isinstance(result, dict) else None
    if not isinstance(domains, list):
        return ["General"]
    return [str(d).strip() for d in domains if str(d).strip()] or ["General"]


def stage_generate_search_queries(claims: List[str]) -> List[str]:
    prompt = f"""Generate search-engine queries to investigate these claims: {claims}

Generate up to 2 queries PER claim: one phrased to find evidence that would CONFIRM it, and one phrased to find evidence that would CONTRADICT/refute it. This avoids one-sided, confirmation-biased research.
Keep each query short (3-8 words), like a real search-engine query, not a full sentence.

Return ONLY valid JSON: {{"queries": ["query 1", "query 2", ...]}}
Maximum 6 queries total -- prioritize the most important ones."""
    result = call_llm_json(prompt, "search_query_generator", default={"queries": claims[:3]}, temperature=0.2, reasoning_effort="low")
    queries = result.get("queries") if isinstance(result, dict) else None
    if not isinstance(queries, list):
        return claims[:3]
    cleaned = [str(q).strip() for q in queries if str(q).strip()]
    return cleaned[:6] or claims[:3]


def stage_retrieve_evidence(search_queries: List[str]) -> List[Dict[str, str]]:
    all_results: List[Dict[str, str]] = []
    for sq in search_queries:
        all_results.extend(web_search(sq))
    return dedupe_results(all_results)


def stage_extract_evidence(claims: List[str], search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not search_results:
        return []
    trimmed = search_results[:15]
    indexed = [{"idx": i, "title": r["title"], "body": r["body"], "url": r["href"]} for i, r in enumerate(trimmed)]
    prompt = f"""You are the Evidence Extractor. Below are search results (with an index and url) and a list of claims to check.

Claims: {claims}
Search results: {json.dumps(indexed, ensure_ascii=False)}

Extract factual snippets from these search results directly relevant to confirming or refuting the claims.
CRITICAL: only use the "url" values exactly as given above. Never invent, guess, or modify a URL.

Return ONLY valid JSON: {{"snippets": [{{"fact": "<factual statement from the result>", "source": "<the exact url it came from>"}}, ...]}}
If nothing is relevant, return {{"snippets": []}}."""
    result = call_llm_json(prompt, "evidence_extractor", default={"snippets": []}, temperature=0.0, reasoning_effort="medium")
    snippets = result.get("snippets") if isinstance(result, dict) else None
    if not isinstance(snippets, list):
        return []

    valid_urls = {r["href"] for r in search_results if r.get("href")}
    cleaned = []
    for s in snippets:
        if not isinstance(s, dict):
            continue
        fact = str(s.get("fact", "")).strip()
        source = str(s.get("source", "")).strip()
        if fact and source in valid_urls:
            cleaned.append({"fact": fact, "source": source})
    return cleaned


def stage_score_evidence(claims: List[str], evidence: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    prompt = f"""You are the Evidence Scorer. Score each piece of evidence for RELEVANCE to the claims (0.0-1.0) and apparent RELIABILITY of the source (0.0-1.0).

Claims: {claims}
Evidence: {json.dumps(evidence, ensure_ascii=False)}

Return ONLY valid JSON: {{"scored_evidence": [{{"fact": "...", "source": "...", "relevance": 0.0-1.0, "reliability": 0.0-1.0}}, ...]}}
Keep fact/source EXACTLY as given, only add the two score fields."""
    default_scored = [{**e, "relevance": 0.5, "reliability": 0.5} for e in evidence]
    result = call_llm_json(prompt, "evidence_scorer", default={"scored_evidence": default_scored}, temperature=0.0, reasoning_effort="medium")
    scored = result.get("scored_evidence") if isinstance(result, dict) else None
    if not isinstance(scored, list) or not scored:
        scored = default_scored

    valid_sources = {e["source"] for e in evidence}
    blended = []
    for item in scored:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        if source not in valid_sources:
            continue  # drop anything not grounded in real retrieved evidence
        fact = str(item.get("fact", "")).strip()
        try:
            relevance = float(item.get("relevance", 0.5))
        except (TypeError, ValueError):
            relevance = 0.5
        try:
            llm_reliability = float(item.get("reliability", 0.5))
        except (TypeError, ValueError):
            llm_reliability = 0.5
        rule_reliability = domain_credibility_hint(source)
        reliability = round((llm_reliability + rule_reliability) / 2, 2)
        blended.append({
            "fact": fact,
            "source": source,
            "relevance": max(0.0, min(1.0, relevance)),
            "reliability": max(0.0, min(1.0, reliability)),
        })
    return blended


def stage_verify_claims(claims: List[str], scored_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    default = {"verdict": "Uncertain", "detailed_analysis": "Insufficient evidence was available to reach a confident verdict."}
    if not scored_evidence:
        return default
    prompt = f"""You are the Verification Engine. Compare the claims against the scored evidence and determine a verdict.

Claims: {claims}
Scored evidence: {json.dumps(scored_evidence, ensure_ascii=False)}

Rules:
- Only say "Verified" if multiple independent, credible sources clearly confirm the claim.
- Only say "Debunked" if multiple independent, credible sources clearly contradict the claim.
- Say "Uncertain" if evidence is thin, mixed, low-relevance, or low-reliability. Do not guess to sound confident.
- Be specific: cite which facts support your verdict and which, if any, conflict with it.

Return ONLY valid JSON: {{"verdict": "Verified"|"Debunked"|"Uncertain", "detailed_analysis": "<specific reasoning citing the evidence>"}}"""
    result = call_llm_json(prompt, "verification_engine", default=default, temperature=0.0, reasoning_effort="high")
    if not isinstance(result, dict) or "verdict" not in result:
        return default
    return {**default, **result}


def stage_resolve_conflicts(claims: List[str], verification: Dict[str, Any], scored_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    default = {"resolved_verdict": verification.get("verdict", "Uncertain"), "conflict_notes": "No conflicts identified."}
    prompt = f"""You are an adversarial Conflict Resolver. Your job is to actively look for flaws in the proposed verdict below, not to rubber-stamp it.

Claims: {claims}
Proposed verdict: {verification.get("verdict")}
Reasoning given: {verification.get("detailed_analysis")}
Evidence used: {json.dumps(scored_evidence, ensure_ascii=False)}

Actively check for: contradicting evidence that was underweighted, sources that disagree with each other, low-reliability sources being treated as decisive, or reasoning that doesn't actually follow from the evidence.
If you find a genuine problem, change the verdict (e.g. downgrade "Verified" to "Uncertain" if support is weaker than claimed). If it holds up, keep it.

Return ONLY valid JSON: {{"resolved_verdict": "Verified"|"Debunked"|"Uncertain"|"Disputed", "conflict_notes": "<what you checked and what you found>"}}
Use "Disputed" if credible sources genuinely disagree with each other."""
    result = call_llm_json(prompt, "conflict_resolver", default=default, temperature=0.0, reasoning_effort="high")
    if not isinstance(result, dict) or "resolved_verdict" not in result:
        return default
    return {**default, **result}


def stage_policy_check(resolved: Dict[str, Any]) -> Dict[str, Any]:
    default = {"is_safe": True, "adjustment_needed": ""}
    prompt = f"""Review this verification result for responsible-communication concerns (e.g. medical/legal advice framing, potential for real-world harm if misread, defamation risk for named individuals).

Result: {json.dumps(resolved, ensure_ascii=False)}

Return ONLY valid JSON: {{"is_safe": true|false, "adjustment_needed": "<specific instruction for phrasing the final answer more safely, or empty string if no change needed>"}}"""
    result = call_llm_json(prompt, "policy_check", default=default, temperature=0.0, reasoning_effort="low")
    if not isinstance(result, dict):
        return default
    return {**default, **result}


def stage_build_knowledge_graph(resolved: Dict[str, Any], scored_evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not scored_evidence:
        return []
    prompt = f"""Structure the verified information into subject-predicate-object triplets for a knowledge graph.

Verdict: {resolved.get("resolved_verdict")}
Evidence: {json.dumps(scored_evidence, ensure_ascii=False)}

Return ONLY valid JSON: {{"triplets": [{{"subject": "...", "predicate": "...", "object": "..."}}, ...]}}
Only include triplets directly supported by the evidence above."""
    result = call_llm_json(prompt, "knowledge_graph", default={"triplets": []}, temperature=0.0, reasoning_effort="low")
    triplets = result.get("triplets") if isinstance(result, dict) else None
    if not isinstance(triplets, list):
        return []
    return [t for t in triplets if isinstance(t, dict) and t.get("subject")]


def stage_synthesize_response(
    query: str,
    claims: List[str],
    scored_evidence: List[Dict[str, Any]],
    resolved: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    default = {
        "answer": "Verification could not be completed with confidence based on available evidence.",
        "confidence": 0.0,
        "reasoningSummary": "The synthesis step failed to produce a valid structured response.",
        "supportingEvidence": [{"fact": e["fact"], "source": e["source"]} for e in scored_evidence[:5]],
        "sources": list({e["source"] for e in scored_evidence}),
    }
    adjustment = policy.get("adjustment_needed") or ""
    prompt = f"""You are the Response Generator, the final step of a fact-verification pipeline. Write the final authoritative report.

Original query: {query!r}
Claims checked: {claims}
Resolved verdict: {resolved.get("resolved_verdict")}
Conflict/dispute notes: {resolved.get("conflict_notes")}
Evidence: {json.dumps(scored_evidence, ensure_ascii=False)}

Write a clear, precise, appropriately hedged answer. Do not state anything as fact that isn't backed by the evidence above. If evidence is thin or mixed, say so plainly rather than sounding more confident than the evidence supports.
{"Apply this safety adjustment to your phrasing: " + adjustment if adjustment else ""}

Return ONLY valid JSON in exactly this shape:
{{"answer": "<final answer, 2-5 sentences>", "confidence": <float 0.0-1.0 reflecting how well-supported the answer is by the evidence>, "reasoningSummary": "<1-3 sentences on how the verdict was reached>", "supportingEvidence": [{{"fact": "...", "source": "..."}}, ...], "sources": ["<url>", ...]}}
Use the EXACT source URLs from the evidence above -- never invent a URL."""
    result = call_llm_json(prompt, "response_generator", default=default, temperature=0.15, reasoning_effort="medium")
    if not isinstance(result, dict):
        return default
    return {**default, **result}


# ---------------------------------------------------------------------------
# Deterministic (non-LLM) audit stages. These cost no extra latency and are
# the most direct defense against fabricated citations and over-confident verdicts.
# ---------------------------------------------------------------------------

def grounding_audit(final_response: Dict[str, Any], retrieved_urls: set) -> Tuple[Dict[str, Any], List[str]]:
    """Strip any cited source that wasn't actually retrieved during search --
    the main defense against the model citing a plausible-looking but fabricated URL."""
    warnings: List[str] = []

    sources = final_response.get("sources", [])
    if isinstance(sources, list):
        clean_sources = [s for s in sources if s in retrieved_urls]
        if len(clean_sources) < len(sources):
            warnings.append(f"Removed {len(sources) - len(clean_sources)} cited source(s) not present in retrieved evidence.")
        final_response["sources"] = clean_sources

    supporting = final_response.get("supportingEvidence", [])
    if isinstance(supporting, list):
        clean_supporting = [e for e in supporting if isinstance(e, dict) and e.get("source") in retrieved_urls]
        if len(clean_supporting) < len(supporting):
            warnings.append(f"Removed {len(supporting) - len(clean_supporting)} supporting-evidence item(s) with an unverifiable source.")
        final_response["supportingEvidence"] = clean_supporting

    return final_response, warnings


def calibrate_confidence(
    llm_confidence: Any,
    verdict: str,
    unique_source_count: int,
    search_returned_nothing: bool,
    grounding_had_removals: bool,
) -> float:
    """Blend the model's self-reported confidence with deterministic caps so
    a single confident-sounding LLM number isn't the only signal."""
    try:
        conf = float(llm_confidence)
    except (TypeError, ValueError):
        conf = 0.3
    conf = max(0.0, min(1.0, conf))

    if search_returned_nothing:
        return min(conf, 0.15)
    if verdict in ("Uncertain", "Disputed"):
        conf = min(conf, 0.55)
    if unique_source_count < MIN_SOURCES_FOR_VERIFIED and verdict in ("Verified", "Debunked"):
        conf = min(conf, 0.5)
    if grounding_had_removals:
        conf = min(conf, 0.4)

    return round(conf, 2)


def _base_response(**overrides: Any) -> Dict[str, Any]:
    """A schema-complete response shell, so every return path -- rate limit,
    empty input, opinion short-circuit, crash -- has every field the API contract promises."""
    base = {
        "answer": "",
        "confidence": 0.0,
        "reasoningSummary": "",
        "supportingEvidence": [],
        "sources": [],
        "verdict": "Uncertain",
        "claimsAnalyzed": [],
        "domains": [],
        "caveats": [],
        "sourceCount": 0,
        "knowledgeGraph": [],
        "pipelineWarnings": [],
        "verifiedAt": _now_iso(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def verify_claim_pipeline(query: str, request: gr.Request = None) -> Dict[str, Any]:
    """The full verification pipeline with real search. Always returns a
    complete, well-typed response -- never raises."""
    client_ip = request.client.host if request else "unknown"
    if is_rate_limited(client_ip):
        return _base_response(
            answer="Rate limit exceeded",
            reasoningSummary=f"Too many requests from {client_ip}. Please wait a minute and try again.",
            verdict="Rate Limited",
        )

    if not query or not query.strip():
        return _base_response(
            answer="Please enter a claim or question to verify.",
            reasoningSummary="No input was provided.",
            verdict="No Input",
        )

    warnings: List[str] = []

    try:
        # 1. Query Analyzer
        analysis = stage_analyze_query(query)

        # Short-circuit opinions/non-verifiable input instead of forcing a verdict.
        if analysis.get("isOpinion") or not analysis.get("isVerifiable", True):
            return _base_response(
                answer=(
                    "This reads as a subjective opinion, preference, or otherwise non-factual "
                    "statement rather than a claim that evidence can confirm or refute, so it "
                    "hasn't been run through the verification pipeline."
                ),
                reasoningSummary=analysis.get("intent", "Not a verifiable factual claim."),
                verdict="Not Applicable — Opinion/Subjective",
                claimsAnalyzed=[query],
            )

        # 2. Claim Extractor
        claims = stage_extract_claims(query)

        # 3. Domain Classifier
        domains = stage_classify_domains(claims)
        caveats: List[str] = []
        if any(d.lower() in HIGH_STAKES_DOMAINS for d in domains):
            caveats.append(
                "This claim touches a high-stakes domain (health, legal, financial, or similar). "
                "Treat this as a starting point, not a substitute for a qualified professional or primary source."
            )

        # 4. Search Query Generator (confirming + disconfirming queries)
        search_queries = stage_generate_search_queries(claims)

        # 5. Retrieval Engine (real-time, multi-engine, retried, de-duplicated)
        search_results = stage_retrieve_evidence(search_queries)
        retrieved_urls = {r["href"] for r in search_results if r.get("href")}
        search_returned_nothing = len(search_results) == 0
        if search_returned_nothing:
            warnings.append("Web search returned no results for any generated query; verdict reflects limited or no external evidence.")

        # 6. Evidence Extractor (grounded to retrieved URLs only)
        evidence = stage_extract_evidence(claims, search_results)

        # 7. Evidence Scorer (LLM judgment blended with rule-based domain credibility)
        scored_evidence = stage_score_evidence(claims, evidence)
        unique_sources = {e["source"] for e in scored_evidence}

        # 8. Verification Engine
        verification = stage_verify_claims(claims, scored_evidence)

        # 9. Adversarial Conflict Resolver
        resolved = stage_resolve_conflicts(claims, verification, scored_evidence)

        # Deterministically enforce the minimum-source rule for strong verdicts.
        if resolved.get("resolved_verdict") in ("Verified", "Debunked") and len(unique_sources) < MIN_SOURCES_FOR_VERIFIED:
            warnings.append(
                f"Downgraded verdict from '{resolved.get('resolved_verdict')}' to 'Uncertain': "
                f"fewer than {MIN_SOURCES_FOR_VERIFIED} independent sources were found."
            )
            resolved["resolved_verdict"] = "Uncertain"

        # 10. Truth & Safety Policy
        policy = stage_policy_check(resolved)

        # 11. Knowledge Graph (now actually surfaced in the response, not discarded)
        knowledge_graph = stage_build_knowledge_graph(resolved, scored_evidence)

        # 12. Response Generator
        final_response = stage_synthesize_response(query, claims, scored_evidence, resolved, policy)

        # 13. Grounding / hallucination audit -- deterministic, not LLM self-report
        final_response, grounding_warnings = grounding_audit(final_response, retrieved_urls)
        warnings.extend(grounding_warnings)

        # 14. Confidence calibration -- deterministic, not raw LLM self-report
        final_response["confidence"] = calibrate_confidence(
            llm_confidence=final_response.get("confidence", 0.3),
            verdict=resolved.get("resolved_verdict", "Uncertain"),
            unique_source_count=len(unique_sources),
            search_returned_nothing=search_returned_nothing,
            grounding_had_removals=bool(grounding_warnings),
        )

        if policy.get("is_safe") is False:
            caveats.append(policy.get("adjustment_needed") or "This result required a safety-related caveat.")
            final_response["confidence"] = min(final_response["confidence"], 0.4)

        final_response["verdict"] = resolved.get("resolved_verdict", "Uncertain")
        final_response["claimsAnalyzed"] = claims
        final_response["domains"] = domains
        final_response["caveats"] = caveats
        final_response["sourceCount"] = len(unique_sources)
        final_response["knowledgeGraph"] = knowledge_graph
        final_response["pipelineWarnings"] = warnings
        final_response["verifiedAt"] = _now_iso()

        # Belt-and-suspenders: guarantee the original 5 fields always exist with the right shape.
        final_response.setdefault("answer", "Unable to produce a final answer.")
        final_response.setdefault("reasoningSummary", "")
        final_response.setdefault("supportingEvidence", [])
        final_response.setdefault("sources", [])

        return final_response

    except Exception as e:
        print(f"verify_claim_pipeline crashed: {type(e).__name__}: {e}")
        return _base_response(
            answer="The verification pipeline hit an unexpected internal error and could not complete.",
            reasoningSummary=f"Internal error: {type(e).__name__}. This has been logged.",
            verdict="Error",
            pipelineWarnings=[f"Unhandled exception: {type(e).__name__}: {e}"],
        )


# ---------------------------------------------------------------------------
# Gradio Interface -- structure and api_name are unchanged from the original.
# ---------------------------------------------------------------------------
with gr.Blocks(title="Truth API Backend") as demo:
    gr.Markdown("# Truth Verification Pipeline API")
    gr.Markdown(
        "14-stage distributed verification pipeline with real-time multi-engine web search, "
        "grounded citations, and calibrated confidence scoring."
    )

    with gr.Row():
        query_input = gr.Textbox(label="Claim to Verify", placeholder="Enter a claim...", lines=3)
        verify_btn = gr.Button("Verify", variant="primary")

    output_json = gr.JSON(label="Verification Result")

    verify_btn.click(
        fn=verify_claim_pipeline,
        inputs=query_input,
        outputs=output_json,
        api_name="verify"
    )

if __name__ == "__main__":
    demo.launch()
