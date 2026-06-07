from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


DEFAULT_DATA_DIR = Path(
    r"D:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
)
RERANK_MODEL = "gemini-2.5-flash"
VECTOR_DIMS = 768

CORE_AI_SKILLS = {
    "python": 10,
    "fastapi": 8,
    "django": 6,
    "flask": 4,
    "llm": 10,
    "llms": 10,
    "rag": 9,
    "retrieval": 9,
    "ranking": 10,
    "recommendation systems": 8,
    "search": 7,
    "embeddings": 9,
    "fine-tuning llms": 8,
    "fine-tuning": 7,
    "langchain": 7,
    "llamaindex": 7,
    "pinecone": 8,
    "chroma": 8,
    "pgvector": 8,
    "milvus": 7,
    "weaviate": 7,
    "vector database": 9,
    "vector databases": 9,
    "machine learning": 7,
    "mlops": 7,
    "bentoml": 6,
    "kubernetes": 5,
    "aws": 4,
    "gcp": 4,
    "postgresql": 4,
    "redis": 3,
    "kafka": 4,
    "airflow": 3,
}

POSITIVE_TERMS = {
    "owned": 8,
    "ownership": 8,
    "from scratch": 10,
    "built": 4,
    "shipped": 7,
    "production": 8,
    "deployed": 6,
    "mentored": 8,
    "mentor": 8,
    "technical leadership": 8,
    "founding": 10,
    "architect": 7,
    "lead": 5,
    "open source": 8,
    "github": 6,
    "maintainer": 9,
    "a/b testing": 6,
    "evaluation": 6,
    "recruiter": 4,
    "matching": 6,
}

NEGATIVE_TERMS = {
    "research-only": 12,
    "academic lab": 10,
    "recent langchain": 8,
    "toy project": 8,
    "no production": 10,
    "maintenance only": 6,
    "frequent switches": 10,
    "short stint": 10,
}


@dataclass
class CandidateScore:
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    semantic_score: float
    experience_score: float
    behavioral_score: float
    tenure_score: float
    name: str


@dataclass
class RuntimeState:
    gemini_enabled: bool
    fallback_reason: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
        text = re.sub(r"<[^>]+>", " ", xml)
        return html.unescape(re.sub(r"\s+", " ", text)).strip()
    except Exception:
        return (
            "Senior AI Engineer, founding team. Needs production ranking, retrieval, embeddings, LLM systems, "
            "scrappy product engineering, mentorship, and stable tenure. Target 5-9 years."
        )


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def hashed_vector(text: str, dims: int = VECTOR_DIMS) -> Dict[int, float]:
    vector: Dict[int, float] = {}
    seen_tokens = set()
    for token in tokenize(text):
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] = vector.get(index, 0.0) + sign * (1.0 + min(len(token), 20) / 20.0)
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        return {index: value / norm for index, value in vector.items()}
    return {0: 1.0}


def dense_hashed_vector(text: str, dims: int = VECTOR_DIMS) -> List[float]:
    vector = [0.0] * dims
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + min(len(token), 20) / 20.0)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        return [value / norm for value in vector]
    vector[0] = 1.0
    return vector


def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(index, 0.0) for index, value in a.items())


def bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def nested_candidate_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile") or {}
    history = candidate.get("career_history") or []
    education = candidate.get("education") or []
    skills = candidate.get("skills") or []
    signals = candidate.get("redrob_signals") or {}

    skill_text = ", ".join(
        f"{skill.get('name', '')} {skill.get('proficiency', '')} {skill.get('duration_months', '')} months"
        for skill in skills
    )
    history_text = " ".join(
        f"{role.get('title', '')} at {role.get('company', '')} for {role.get('duration_months', 0)} months. "
        f"{role.get('description', '')}"
        for role in history
    )
    education_text = " ".join(
        f"{edu.get('degree', '')} {edu.get('field_of_study', '')} {edu.get('tier', '')}"
        for edu in education
    )
    signal_text = (
        f"github {signals.get('github_activity_score', -1)} response {signals.get('recruiter_response_rate', 0)} "
        f"interview {signals.get('interview_completion_rate', 0)} saved {signals.get('saved_by_recruiters_30d', 0)} "
        f"notice {signals.get('notice_period_days', 0)} open_to_work {signals.get('open_to_work_flag', False)}"
    )
    return clean_text(
        " ".join(
            [
                profile.get("anonymized_name", ""),
                profile.get("headline", ""),
                profile.get("summary", ""),
                profile.get("current_title", ""),
                profile.get("current_industry", ""),
                skill_text,
                history_text,
                education_text,
                signal_text,
            ]
        )
    )


def skill_score(candidate: Dict[str, Any]) -> float:
    skills = candidate.get("skills") or []
    profile = candidate.get("profile") or {}
    text = " ".join(
        [
            profile.get("headline", ""),
            profile.get("summary", ""),
            " ".join(skill.get("name", "") for skill in skills),
        ]
    ).lower()
    raw = 0.0
    for term, weight in CORE_AI_SKILLS.items():
        if term in text:
            raw += weight
    for skill in skills:
        name = str(skill.get("name", "")).lower()
        proficiency = str(skill.get("proficiency", "")).lower()
        duration = float(skill.get("duration_months") or 0)
        if name in CORE_AI_SKILLS:
            raw += CORE_AI_SKILLS[name] * {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.1}.get(
                proficiency, 0.6
            )
            raw += min(duration / 12.0, 5.0)
    return bounded(raw)


def experience_score(candidate: Dict[str, Any]) -> float:
    years = float((candidate.get("profile") or {}).get("years_of_experience") or 0.0)
    if 5.0 <= years <= 9.0:
        return 100.0
    if years < 5.0:
        return bounded((years / 5.0) * 100.0)
    return bounded(100.0 - min((years - 9.0) * 5.0, 35.0))


def behavioral_score(candidate: Dict[str, Any], full_text: str) -> float:
    signals = candidate.get("redrob_signals") or {}
    score = 35.0
    lower = full_text.lower()

    for term, weight in POSITIVE_TERMS.items():
        if term in lower:
            score += weight
    for term, weight in NEGATIVE_TERMS.items():
        if term in lower:
            score -= weight

    github = float(signals.get("github_activity_score") or 0.0)
    if github >= 0:
        score += min(github, 100.0) * 0.12
    score += float(signals.get("recruiter_response_rate") or 0.0) * 10.0
    score += float(signals.get("interview_completion_rate") or 0.0) * 8.0
    score += min(float(signals.get("saved_by_recruiters_30d") or 0.0), 20.0) * 0.6
    if signals.get("open_to_work_flag"):
        score += 4.0
    if signals.get("verified_email"):
        score += 1.5
    if signals.get("verified_phone"):
        score += 1.5
    if signals.get("linkedin_connected"):
        score += 2.0
    return bounded(score)


def tenure_score(candidate: Dict[str, Any]) -> float:
    history = candidate.get("career_history") or []
    if not history:
        return 40.0
    months = [float(role.get("duration_months") or 0.0) for role in history if float(role.get("duration_months") or 0.0) > 0]
    if not months:
        return 40.0
    avg_years = statistics.mean(months) / 12.0
    short_stints = sum(1 for month in months if month < 18)
    score = bounded((avg_years / 2.0) * 100.0)
    score -= short_stints * 12.0
    return bounded(score)


def location_score(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    country = str(profile.get("country", "")).lower()
    location = str(profile.get("location", "")).lower()
    tier_one = {"pune", "noida", "delhi", "gurugram", "gurgaon", "bangalore", "bengaluru", "mumbai", "hyderabad", "chennai"}
    if "india" in country and any(city in location for city in tier_one):
        return 100.0
    if "india" in country:
        return 85.0 if signals.get("willing_to_relocate") else 70.0
    return 55.0 if signals.get("willing_to_relocate") else 35.0


def score_candidate(candidate: Dict[str, Any], jd_vector: Dict[int, float]) -> CandidateScore:
    cid = str(candidate.get("candidate_id", ""))
    profile = candidate.get("profile") or {}
    name = str(profile.get("anonymized_name") or cid)
    text = nested_candidate_text(candidate)
    semantic = bounded((cosine(jd_vector, hashed_vector(text)) + 1.0) * 50.0)
    skill = skill_score(candidate)
    experience = experience_score(candidate)
    behavior = behavioral_score(candidate, text)
    tenure = tenure_score(candidate)
    location = location_score(candidate)

    score = (
        0.25 * semantic
        + 0.25 * skill
        + 0.20 * experience
        + 0.15 * behavior
        + 0.10 * tenure
        + 0.05 * location
    )

    if skill < 25:
        score -= 12.0
    if experience < 65:
        score -= 8.0
    if tenure < 55:
        score -= 6.0

    top_skills = sorted(
        [skill.get("name", "") for skill in candidate.get("skills") or [] if str(skill.get("name", "")).lower() in CORE_AI_SKILLS],
        key=lambda item: CORE_AI_SKILLS.get(item.lower(), 0),
        reverse=True,
    )[:5]
    years = float(profile.get("years_of_experience") or 0.0)
    reasoning = (
        f"{profile.get('current_title', 'Candidate')} with {years:.1f} yrs; "
        f"AI/backend match {skill:.0f}/100 via {', '.join(top_skills) if top_skills else 'limited core skills'}; "
        f"behavior {behavior:.0f}, tenure {tenure:.0f}."
    )

    return CandidateScore(
        candidate_id=cid,
        rank=0,
        score=bounded(score) / 100.0,
        reasoning=reasoning[:450],
        semantic_score=semantic,
        experience_score=experience,
        behavioral_score=behavior,
        tenure_score=tenure,
        name=name,
    )


def iter_json_candidates(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and "candidates" in data:
            yield from data["candidates"]
        else:
            yield data


def keep_top(scores: List[CandidateScore], candidate_score: CandidateScore, limit: int) -> List[CandidateScore]:
    scores.append(candidate_score)
    scores.sort(key=lambda item: (-item.score, item.candidate_id))
    if len(scores) > limit:
        scores.pop()
    return scores


def create_gemini_client() -> Tuple[Optional[Any], RuntimeState]:
    if genai is None:
        return None, RuntimeState(False, "google-genai is not installed; using deterministic local refinement.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, RuntimeState(False, "GEMINI_API_KEY is not set; using deterministic local refinement.")
    try:
        return genai.Client(api_key=api_key), RuntimeState(True, "")
    except Exception as exc:
        return None, RuntimeState(False, f"Gemini client initialization failed: {exc}")


def rerank_top_15_with_gemini(scores: List[CandidateScore], jd_text: str, state: RuntimeState) -> List[CandidateScore]:
    client, created_state = create_gemini_client()
    if not created_state.gemini_enabled:
        state.gemini_enabled = False
        state.fallback_reason = created_state.fallback_reason
        return scores

    try:
        payload = [
            {
                "candidate_id": item.candidate_id,
                "current_rank": index + 1,
                "score": round(item.score, 4),
                "reasoning": item.reasoning,
                "semantic": round(item.semantic_score, 2),
                "experience": round(item.experience_score, 2),
                "behavior": round(item.behavioral_score, 2),
                "tenure": round(item.tenure_score, 2),
            }
            for index, item in enumerate(scores[:15])
        ]
        prompt = (
            "Rerank these 15 candidates for the Redrob Senior AI Engineer founding-team role. "
            "Prefer production ML/ranking/retrieval depth, backend shipping ability, mentorship, open-source/GitHub signal, "
            "and stable tenure. Return strict JSON with key rankings: list of {candidate_id, rank, reasoning}. "
            "Reasoning must be one concise sentence under 35 words.\n\n"
            f"JD:\n{jd_text[:5000]}\n\nCANDIDATES:\n{json.dumps(payload, indent=2)}"
        )
        response = client.models.generate_content(
            model=RERANK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.15) if types else None,
        )
        text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        rankings = parsed.get("rankings", parsed)
        by_id = {item.candidate_id: item for item in scores}
        ordered: List[CandidateScore] = []
        for item in sorted(rankings, key=lambda value: int(value.get("rank", 999))):
            cid = str(item.get("candidate_id", ""))
            if cid in by_id:
                by_id[cid].reasoning = clean_text(item.get("reasoning")) or by_id[cid].reasoning
                ordered.append(by_id[cid])
        ordered.extend([item for item in scores[:15] if item.candidate_id not in {candidate.candidate_id for candidate in ordered}])
        return ordered + scores[15:]
    except Exception as exc:
        state.gemini_enabled = False
        state.fallback_reason = f"Gemini rerank failed; using local order: {exc}"
        return scores


def assign_final_scores(scores: List[CandidateScore]) -> List[CandidateScore]:
    scores.sort(key=lambda item: (-item.score, item.candidate_id))
    for index, item in enumerate(scores, start=1):
        item.rank = index
        item.score = round(1.0 - ((index - 1) * 0.0065), 4)
    return scores


def write_challenge_csv(scores: Sequence[CandidateScore], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for item in scores:
            writer.writerow([item.candidate_id, item.rank, f"{item.score:.4f}", item.reasoning])


def write_detailed_csv(scores: Sequence[CandidateScore], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Candidate_ID",
                "Full_Name",
                "Semantic_Score",
                "Experience_Score",
                "LLM_Fit_Analysis",
                "Final_Rank",
            ]
        )
        for item in scores:
            writer.writerow(
                [
                    item.candidate_id,
                    item.name,
                    f"{item.semantic_score:.2f}",
                    f"{item.experience_score:.2f}",
                    item.reasoning,
                    item.rank,
                ]
            )


def print_report(scores: Sequence[CandidateScore], challenge_csv: Path, detailed_csv: Path, state: RuntimeState) -> None:
    print("\n# Top 5 Shortlisted Candidates\n")
    print(f"Challenge CSV: `{challenge_csv}`")
    print(f"Detailed CSV: `{detailed_csv}`\n")
    if not state.gemini_enabled:
        print(f"> Fallback note: {state.fallback_reason}\n")
    for item in scores[:5]:
        print(f"## Rank {item.rank}: {item.name} ({item.candidate_id})")
        print(
            f"- Score: **{item.score:.4f}** | Semantic: **{item.semantic_score:.1f}** | "
            f"Experience: **{item.experience_score:.1f}** | Behavior: **{item.behavioral_score:.1f}** | "
            f"Tenure: **{item.tenure_score:.1f}**"
        )
        print(f"- Justification: {item.reasoning}\n")


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    candidates_path = Path(args.candidates) if args.candidates else data_dir / "candidates.jsonl"
    jd_path = Path(args.jd_docx) if args.jd_docx else data_dir / "job_description.docx"
    jd_text = args.jd_text or read_docx_text(jd_path)
    jd_vector = hashed_vector(jd_text)

    top_limit = max(100, args.pool)
    top_scores: List[CandidateScore] = []
    started = time.time()
    for index, candidate in enumerate(iter_json_candidates(candidates_path), start=1):
        top_scores = keep_top(top_scores, score_candidate(candidate, jd_vector), top_limit)
        if args.progress and index % args.progress == 0:
            elapsed = time.time() - started
            print(f"Processed {index:,} candidates in {elapsed:.1f}s; current cutoff {top_scores[-1].score:.4f}")

    top_scores.sort(key=lambda item: (-item.score, item.candidate_id))
    state = RuntimeState(True, "")
    top_scores = rerank_top_15_with_gemini(top_scores[:100], jd_text, state)
    final_scores = assign_final_scores(top_scores[:100])

    challenge_csv = Path(args.challenge_output)
    detailed_csv = Path(args.detailed_output)
    write_challenge_csv(final_scores, challenge_csv)
    write_detailed_csv(final_scores, detailed_csv)
    print_report(final_scores, challenge_csv, detailed_csv, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank India Runs / Redrob challenge candidates.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Challenge data directory.")
    parser.add_argument("--candidates", help="Path to candidates.jsonl or sample_candidates.json.")
    parser.add_argument("--jd-docx", help="Path to job_description.docx.")
    parser.add_argument("--jd-text", help="Override JD text.")
    parser.add_argument("--challenge-output", default="redrob_submission.csv", help="Validator-ready 100-row CSV.")
    parser.add_argument("--detailed-output", default="ranked_candidates.csv", help="Detailed CSV matching the prompt deliverable.")
    parser.add_argument("--pool", type=int, default=250, help="Internal top candidate pool size before final 100.")
    parser.add_argument("--progress", type=int, default=1000, help="Print progress every N candidates; 0 disables it.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
