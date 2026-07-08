https://i.ibb.co/PsycpCXP/Chat-GPT-Image-Jul-7-2026-11-28-32-AM.png

# Truth Verification Pipeline

A multi-stage fact-verification API. Given a claim or question, it extracts
the underlying factual assertions, searches the live web for evidence,
scores and cross-checks that evidence, and returns a verdict with a
calibrated confidence score and the exact sources used.

It's built to be called programmatically (see **API usage** below) as much
as used through the demo UI above.

## Setup

This Space needs two secrets set under **Settings → Repository secrets**:

| Secret | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes (primary) | Runs the pipeline's reasoning/extraction stages via Groq (`openai/gpt-oss-120b`) |
| `GEMINI_API_KEY` | Recommended (fallback) | Used only if Groq is unavailable, via `gemini-3.1-flash-lite` |

If neither key is set, the API responds with a clear error instead of
crashing. If only one is set, that provider handles every stage.

No search API key is needed — evidence retrieval uses `ddgs`, a free
multi-engine metasearch library (no signup required).

## API usage

The UI above exposes a single endpoint, `verify`, via the Gradio client:

```python
from gradio_client import Client

client = Client("your-username/your-space-name")
result = client.predict(
    "The Great Wall of China is visible from space with the naked eye.",
    api_name="/verify"
)
print(result)
```

Or over plain HTTP:

```bash
curl -X POST https://your-username-your-space-name.hf.space/call/verify \
  -H "Content-Type: application/json" \
  -d '{"data": ["Is the Great Wall of China visible from space?"]}'
```

### Response shape

```json
{
  "answer": "string — the final, evidence-grounded answer",
  "confidence": 0.0,
  "reasoningSummary": "string — why this verdict was reached",
  "supportingEvidence": [{"fact": "string", "source": "url"}],
  "sources": ["url", "..."],

  "verdict": "Verified | Debunked | Uncertain | Disputed | Not Applicable — Opinion/Subjective | Rate Limited | No Input | Error",
  "claimsAnalyzed": ["the atomic claims extracted from your input"],
  "domains": ["subject-matter tags, e.g. Science, Health, Politics"],
  "caveats": ["warnings, e.g. high-stakes domain or safety adjustment"],
  "sourceCount": 0,
  "knowledgeGraph": [{"subject": "...", "predicate": "...", "object": "..."}],
  "pipelineWarnings": ["transparency notes about anything that degraded, e.g. a stripped fabricated source"],
  "verifiedAt": "ISO 8601 timestamp"
}
```

`answer`, `confidence`, `reasoningSummary`, `supportingEvidence`, and
`sources` are the original fields and are always present with these exact
names/types. Everything else is additive.

Rate limit: 5 requests per client IP per 60 seconds.

## How the pipeline works

Every input goes through the same 14 stages. Stages 1–12 are LLM calls with
a specific, narrow job; stages 13–14 are plain Python — no model involved —
which is deliberate: the parts of the pipeline that most directly police
accuracy don't rely on an LLM grading its own homework.

1. **Query Analyzer** — classifies the input as a verifiable factual claim
   vs. an opinion/preference. Opinions short-circuit immediately with a
   `Not Applicable — Opinion/Subjective` verdict instead of being forced
   through a Verified/Debunked judgment they were never suited for.
2. **Claim Extractor** — splits the input into distinct, atomic,
   independently-checkable claims (a compound sentence can assert several
   things at once; each is checked on its own).
3. **Domain Classifier** — tags the subject domain(s) (Science, Health,
   Politics, Finance, etc.). Claims in high-stakes domains (health, legal,
   financial, elections) get an explicit caveat appended to the final
   answer.
4. **Search Query Generator** — writes real search-engine queries for each
   claim. Deliberately generates queries aimed at *confirming* the claim
   **and** queries aimed at *refuting* it, to avoid one-sided,
   confirmation-biased research.
5. **Retrieval Engine** — runs those queries against `ddgs`, which itself
   fans out across multiple engines (Bing, Brave, Google, DuckDuckGo,
   Yahoo, Yandex, Wikipedia) and aggregates/de-duplicates results. Retried
   with backoff on transient failures.
6. **Evidence Extractor** — pulls factual snippets out of the raw search
   results. The model is only shown the URLs actually returned by search
   and is explicitly told never to invent one; anything it cites that
   doesn't match a real retrieved URL is discarded right here.
7. **Evidence Scorer** — scores each snippet for relevance and reliability.
   Reliability is a *blend* of the model's judgment and a rule-based prior
   (`.gov`/`.edu`/major wire services score higher, personal blogs score
   lower) — so one model's guess about source quality isn't the only
   signal.
8. **Verification Engine** — reaches a first-pass verdict (Verified /
   Debunked / Uncertain) strictly from the scored evidence, with explicit
   instructions not to guess confidently when evidence is thin or mixed.
9. **Conflict Resolver** — an adversarial second pass whose only job is to
   find problems with the verdict from stage 8: underweighted
   contradictions, sources disagreeing with each other, low-reliability
   evidence being treated as decisive. It can downgrade or change the
   verdict, or mark it "Disputed" if credible sources genuinely disagree.
10. **Truth & Safety Policy** — reviews the resolved verdict for
    responsible-communication concerns (e.g. medical/legal framing,
    defamation risk) and can attach a phrasing adjustment that the final
    answer is required to apply.
11. **Knowledge Graph Builder** — structures the verified facts into
    subject–predicate–object triplets, surfaced in `knowledgeGraph`.
12. **Response Generator** — writes the final answer, reasoning summary,
    and cites its supporting evidence, using only the exact source URLs
    already validated in earlier stages.
13. **Grounding Audit** *(code, not a model call)* — a hard check that
    strips any source in the final answer that isn't in the set of URLs
    actually retrieved in stage 5. This is the last line of defense against
    a fabricated citation slipping through.
14. **Confidence Calibration** *(code, not a model call)* — recomputes the
    final confidence score from concrete signals rather than trusting the
    model's self-reported number: it's capped hard if search returned
    nothing, capped further if fewer than 2 independent sources back a
    "Verified"/"Debunked" verdict, and capped again if the grounding audit
    had to remove anything.

Throughout, every LLM call is wrapped in JSON-repair logic (strip markdown
fences → parse → extract the first balanced JSON block → one corrective
retry) with a typed fallback default, and the whole pipeline runs inside a
top-level try/except — so a single stage failing degrades that one part of
the result (and is logged in `pipelineWarnings`) instead of crashing the
request.

## Known limitations

- Web search quality depends on what the underlying engines surface; a
  claim about something very recent or very obscure may return little or
  no evidence, in which case the pipeline correctly reports low confidence
  rather than guessing.
- The rule-based source-credibility list is a small, illustrative set of
  high/low-credibility domain hints, not an exhaustive authority ranking.
- This is a research/demo tool, not a substitute for professional
  judgment — see the caveats field for claims in high-stakes domains.
