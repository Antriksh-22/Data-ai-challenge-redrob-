from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


EMBEDDING_MODEL = "text-embedding-004"
RERANK_MODEL = "gemini-2.5-flash"

DEFAULT_JOB_DESCRIPTION = (
    "Role: Senior AI & Backend Engineer. Context: Building next-gen automation. "
    "Requirements: 5+ years of Python (FastAPI/Django). LLM pipelines "
    "(LangChain/LlamaIndex). Vector databases (Chroma/Pinecone/pgvector). "
    "Behavioral: Strong open-source contributions, ownership from scratch, "
    "technical mentorship. No job-hoppers (average 2+ years per role)."
)

POSITIVE_BEHAVIOR_KEYWORDS = {
    "open-source",
    "open source",
    "github",
    "maintainer",
    "mentor",
    "mentorship",
    "ownership",
    "from scratch",
    "founding",
    "lead",
    "principal",
    "staff",
    "architect",
    "technical leadership",
    "automation",
    "contributor",
    "oss",
}

NEGATIVE_BEHAVIOR_KEYWORDS = {
    "job hopper",
    "job-hopper",
    "short stint",
    "frequent switches",
    "underperformance",
    "no mentorship",
    "maintenance only",
}


class JobRequirements(BaseModel):
    role_title: str = "Senior AI & Backend Engineer"
    target_years: float = 5.0
    must_have_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    behavioral_signals: List[str] = Field(default_factory=list)
    tenure_requirement_years: float = 2.0
    domain_context: str = "next-gen automation"


class CandidateProfile(BaseModel):
    candidate_id: str
    full_name: str
    skills: str
    experience_years: float
    history: str
    behavior: str

    @property
    def profile_text(self) -> str:
        return " ".join(
            [
                f"Name: {self.full_name}.",
                f"Skills: {self.skills}.",
                f"Experience: {self.experience_years} years.",
                f"History: {self.history}.",
                f"Behavior: {self.behavior}.",
            ]
        )


class EvaluationResult(BaseModel):
    Candidate_ID: str
    Full_Name: str
    Semantic_Score: float
    Experience_Score: float
    Behavioral_Score: float
    Tenure_Score: float
    Hybrid_Score: float
    LLM_Fit_Analysis: str
    Final_Rank: int


@dataclass
class RuntimeState:
    gemini_enabled: bool = True
    fallback_reason: str = ""

    def disable_gemini(self, reason: str) -> None:
        self.gemini_enabled = False
        self.fallback_reason = reason


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def contains_any(text: str, keywords: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def deterministic_vector(text: str, dimensions: int = 768) -> List[float]:
    tokens = re.findall(r"[a-z0-9+#.]+", text.lower())
    vector = np.zeros(dimensions, dtype=np.float32)
    if not tokens:
        vector[0] = 1.0
        return vector.tolist()

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + min(len(token), 18) / 18.0
        vector[index] += sign * weight

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


class JobDescriptionAnalyzer:
    def __init__(self, client: Any, state: RuntimeState) -> None:
        self.client = client
        self.state = state

    def analyze(self, jd_text: str) -> JobRequirements:
        if self.client is not None and self.state.gemini_enabled:
            try:
                prompt = (
                    "Extract structured hiring criteria from this job description. "
                    "Return strict JSON only with keys: role_title, target_years, "
                    "must_have_skills, preferred_skills, behavioral_signals, "
                    "tenure_requirement_years, domain_context.\n\n"
                    f"JOB DESCRIPTION:\n{jd_text}"
                )
                response = self.client.models.generate_content(
                    model=RERANK_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    )
                    if types
                    else None,
                )
                parsed = self._parse_json(response.text)
                return JobRequirements(**parsed)
            except Exception as exc:
                self.state.disable_gemini(f"JD analysis switched to heuristic mode: {exc}")

        return self._heuristic_analysis(jd_text)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
        return json.loads(cleaned)

    @staticmethod
    def _heuristic_analysis(jd_text: str) -> JobRequirements:
        lower = jd_text.lower()
        years_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", lower)
        target_years = float(years_match.group(1)) if years_match else 5.0

        skill_terms = [
            "python",
            "fastapi",
            "django",
            "langchain",
            "llamaindex",
            "chroma",
            "pinecone",
            "pgvector",
            "vector databases",
            "llm pipelines",
            "backend",
            "automation",
        ]
        must_have = [term for term in skill_terms if term in lower]
        behavioral = [kw for kw in POSITIVE_BEHAVIOR_KEYWORDS if kw in lower]

        return JobRequirements(
            role_title="Senior AI & Backend Engineer",
            target_years=target_years,
            must_have_skills=must_have or ["python", "backend", "llm pipelines"],
            preferred_skills=["FastAPI", "Django", "LangChain", "LlamaIndex", "Chroma", "Pinecone", "pgvector"],
            behavioral_signals=behavioral or ["ownership", "open-source", "technical mentorship"],
            tenure_requirement_years=2.0,
            domain_context="next-gen automation",
        )


class EmbeddingEngine:
    def __init__(self, client: Any, state: RuntimeState, batch_size: int = 16, delay_seconds: float = 0.25) -> None:
        self.client = client
        self.state = state
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        if self.client is not None and self.state.gemini_enabled:
            try:
                vectors: List[List[float]] = []
                for start in range(0, len(texts), self.batch_size):
                    batch = list(texts[start : start + self.batch_size])
                    response = self.client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                        if types
                        else None,
                    )
                    vectors.extend([embedding.values for embedding in response.embeddings])
                    if start + self.batch_size < len(texts):
                        time.sleep(self.delay_seconds)
                return np.array(vectors, dtype=np.float32)
            except Exception as exc:
                self.state.disable_gemini(f"Embedding switched to deterministic local vectors: {exc}")

        return np.array([deterministic_vector(text) for text in texts], dtype=np.float32)


class CandidateEvaluator:
    COLUMN_ALIASES = {
        "candidate_id": ["candidate_id", "id", "uid"],
        "full_name": ["name", "full_name", "candidate_name"],
        "skills": ["skills", "technologies", "core_competencies"],
        "experience_years": ["experience", "years_of_experience", "exp_years"],
        "history": ["resume_text", "summary", "past_roles", "description", "history"],
        "behavior": ["platform_activity", "github_activity", "behavioral_signals"],
    }

    def __init__(self, requirements: JobRequirements, embedding_engine: EmbeddingEngine) -> None:
        self.requirements = requirements
        self.embedding_engine = embedding_engine

    def load_candidates(self, csv_path: Optional[str]) -> pd.DataFrame:
        if csv_path:
            path = Path(csv_path)
            if path.exists():
                return pd.read_csv(path)
            print(f"Input file not found at {path}; using generated mock candidates.")
        return generate_mock_dataset()

    def normalize_candidates(self, raw_df: pd.DataFrame) -> List[CandidateProfile]:
        bound = self._bind_columns(raw_df)
        profiles: List[CandidateProfile] = []

        for idx, row in raw_df.iterrows():
            payload = {
                "candidate_id": normalize_text(row.get(bound.get("candidate_id", ""), f"CAND-{idx + 1:03d}"))
                or f"CAND-{idx + 1:03d}",
                "full_name": normalize_text(row.get(bound.get("full_name", ""), f"Candidate {idx + 1}"))
                or f"Candidate {idx + 1}",
                "skills": normalize_text(row.get(bound.get("skills", ""), "")),
                "experience_years": safe_float(row.get(bound.get("experience_years", ""), 0.0)),
                "history": normalize_text(row.get(bound.get("history", ""), "")),
                "behavior": normalize_text(row.get(bound.get("behavior", ""), "")),
            }
            try:
                profiles.append(CandidateProfile(**payload))
            except ValidationError:
                payload["experience_years"] = 0.0
                profiles.append(CandidateProfile(**payload))

        return profiles

    def score(self, profiles: Sequence[CandidateProfile], jd_text: str) -> pd.DataFrame:
        jd_vector = self.embedding_engine.embed_texts([jd_text])
        candidate_vectors = self.embedding_engine.embed_texts([profile.profile_text for profile in profiles])
        semantic_raw = cosine_similarity(jd_vector, candidate_vectors)[0]
        semantic_scores = np.clip((semantic_raw + 1.0) / 2.0, 0.0, 1.0) * 100.0

        rows: List[Dict[str, Any]] = []
        for index, profile in enumerate(profiles):
            experience_score = self._experience_score(profile.experience_years)
            behavioral_score = self._behavioral_score(profile)
            tenure_score = self._tenure_score(profile)
            hybrid = (
                0.40 * semantic_scores[index]
                + 0.30 * experience_score
                + 0.20 * behavioral_score
                + 0.10 * tenure_score
            )

            rows.append(
                {
                    "Candidate_ID": profile.candidate_id,
                    "Full_Name": profile.full_name,
                    "Semantic_Score": round(float(semantic_scores[index]), 2),
                    "Experience_Score": round(experience_score, 2),
                    "Behavioral_Score": round(behavioral_score, 2),
                    "Tenure_Score": round(tenure_score, 2),
                    "Hybrid_Score": round(hybrid, 2),
                    "Profile_Text": profile.profile_text,
                    "Experience_Years": profile.experience_years,
                }
            )

        return pd.DataFrame(rows).sort_values("Hybrid_Score", ascending=False).reset_index(drop=True)

    def _bind_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        normalized_columns = {str(column).strip().lower(): column for column in df.columns}
        bound: Dict[str, str] = {}
        for canonical, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias.lower() in normalized_columns:
                    bound[canonical] = normalized_columns[alias.lower()]
                    break
        return bound

    def _experience_score(self, years: float) -> float:
        required = max(self.requirements.target_years, 0.1)
        return float(np.clip((years / required) * 100.0, 0.0, 100.0))

    def _behavioral_score(self, profile: CandidateProfile) -> float:
        text = f"{profile.skills} {profile.history} {profile.behavior}".lower()
        positives = contains_any(text, POSITIVE_BEHAVIOR_KEYWORDS)
        negatives = contains_any(text, NEGATIVE_BEHAVIOR_KEYWORDS)
        score = 45.0 + positives * 10.0 - negatives * 15.0

        for signal in self.requirements.behavioral_signals:
            if signal.lower() in text:
                score += 5.0

        return float(np.clip(score, 0.0, 100.0))

    def _tenure_score(self, profile: CandidateProfile) -> float:
        text = f"{profile.history} {profile.behavior}".lower()
        durations = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:yr|yrs|year|years)", text)]
        if durations:
            average_tenure = sum(durations) / len(durations)
        else:
            role_count = max(1, len(re.findall(r"\b(?:engineer|developer|lead|architect|manager|consultant)\b", text)))
            average_tenure = profile.experience_years / role_count if profile.experience_years else 0.0

        if average_tenure >= self.requirements.tenure_requirement_years:
            return 100.0
        if average_tenure < 1.5:
            return float(np.clip((average_tenure / 1.5) * 70.0, 0.0, 70.0))
        return float(np.clip((average_tenure / self.requirements.tenure_requirement_years) * 100.0, 0.0, 100.0))


class GeminiReranker:
    def __init__(self, client: Any, state: RuntimeState, requirements: JobRequirements) -> None:
        self.client = client
        self.state = state
        self.requirements = requirements

    def rerank(self, top_df: pd.DataFrame, jd_text: str) -> pd.DataFrame:
        working = top_df.copy()
        if self.client is not None and self.state.gemini_enabled:
            try:
                payload = working[
                    [
                        "Candidate_ID",
                        "Full_Name",
                        "Semantic_Score",
                        "Experience_Score",
                        "Behavioral_Score",
                        "Tenure_Score",
                        "Hybrid_Score",
                        "Profile_Text",
                    ]
                ].to_dict(orient="records")
                prompt = self._build_prompt(jd_text, payload)
                response = self.client.models.generate_content(
                    model=RERANK_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    )
                    if types
                    else None,
                )
                refined = self._parse_refinement(response.text)
                return self._merge_refinement(working, refined)
            except Exception as exc:
                self.state.disable_gemini(f"Reranking switched to heuristic mode: {exc}")

        return self._heuristic_refinement(working)

    def _build_prompt(self, jd_text: str, candidates: List[Dict[str, Any]]) -> str:
        return (
            "You are ranking exactly 15 candidates for a Senior AI & Backend Engineer role. "
            "Use the math metrics as evidence, but refine qualitatively for LLM/backend depth, "
            "ownership, mentorship, open-source signal, and tenure stability. Return strict JSON "
            "with key 'rankings', a list of objects containing Candidate_ID, rank, and analysis. "
            "Each analysis must be exactly two concise sentences.\n\n"
            f"JOB DESCRIPTION:\n{jd_text}\n\n"
            f"STRUCTURED REQUIREMENTS:\n{self.requirements.model_dump_json()}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, ensure_ascii=True, indent=2)}"
        )

    @staticmethod
    def _parse_refinement(text: str) -> List[Dict[str, Any]]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        rankings = data["rankings"] if isinstance(data, dict) and "rankings" in data else data
        return list(rankings)

    @staticmethod
    def _merge_refinement(df: pd.DataFrame, refined: List[Dict[str, Any]]) -> pd.DataFrame:
        analysis_by_id = {
            str(item.get("Candidate_ID")): normalize_text(item.get("analysis", ""))
            for item in refined
        }
        rank_by_id = {
            str(item.get("Candidate_ID")): int(item.get("rank", 999))
            for item in refined
            if str(item.get("rank", "")).isdigit()
        }
        working = df.copy()
        working["LLM_Fit_Analysis"] = working["Candidate_ID"].map(analysis_by_id).fillna(
            "Strong mathematical fit, but the structured reranker returned limited narrative evidence. "
            "Review the profile for final human calibration."
        )
        working["Final_Rank"] = working["Candidate_ID"].map(rank_by_id).fillna(999).astype(int)
        working = working.sort_values(["Final_Rank", "Hybrid_Score"], ascending=[True, False]).reset_index(drop=True)
        working["Final_Rank"] = range(1, len(working) + 1)
        return working

    @staticmethod
    def _heuristic_refinement(df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy().sort_values(
            ["Hybrid_Score", "Behavioral_Score", "Tenure_Score"], ascending=False
        )
        analyses = []
        for _, row in working.iterrows():
            strengths = []
            if row["Semantic_Score"] >= 70:
                strengths.append("strong semantic alignment")
            if row["Experience_Score"] >= 95:
                strengths.append("senior-level experience")
            if row["Behavioral_Score"] >= 75:
                strengths.append("clear ownership and mentorship signals")
            if row["Tenure_Score"] >= 90:
                strengths.append("stable role tenure")
            if not strengths:
                strengths.append("a balanced but not dominant profile")
            sentence_one = f"{row['Full_Name']} shows {', '.join(strengths[:3])} for the automation-heavy AI backend role."
            sentence_two = (
                "The fallback reranker favors the candidate's combined math score while preserving penalties for weak "
                "behavioral evidence or short-tenure patterns."
            )
            analyses.append(f"{sentence_one} {sentence_two}")
        working["LLM_Fit_Analysis"] = analyses
        working["Final_Rank"] = range(1, len(working) + 1)
        return working.reset_index(drop=True)


def generate_mock_dataset() -> pd.DataFrame:
    random.seed(42)
    first_names = [
        "Aarav",
        "Isha",
        "Rohan",
        "Meera",
        "Kabir",
        "Ananya",
        "Dev",
        "Tara",
        "Nikhil",
        "Zoya",
    ]
    last_names = [
        "Sharma",
        "Rao",
        "Mehta",
        "Kapoor",
        "Iyer",
        "Nair",
        "Saxena",
        "Menon",
        "Bose",
        "Khan",
    ]
    archetypes = [
        {
            "skills": "Python, FastAPI, Django, LangChain, LlamaIndex, pgvector, Pinecone, Chroma, Kubernetes",
            "history": "Senior Backend Engineer 3.2 years, AI Platform Lead 2.8 years, built LLM automation from scratch",
            "behavior": "GitHub open-source maintainer, technical mentorship, strong ownership, architecture reviews",
            "years": (6.0, 9.5),
        },
        {
            "skills": "Python, Django, REST APIs, PostgreSQL, Celery, Redis, AWS",
            "history": "Backend Developer 2.4 years, Senior Engineer 2.1 years, API platform modernization",
            "behavior": "Reliable owner, limited LLM exposure, mentors junior engineers",
            "years": (4.5, 7.0),
        },
        {
            "skills": "Python, LangChain, prompt engineering, Chroma, Streamlit, OpenAI APIs",
            "history": "ML Engineer 1.1 years, AI Engineer 1.3 years, consultant 0.9 years",
            "behavior": "Frequent switches, strong demos, limited production ownership",
            "years": (2.5, 4.0),
        },
        {
            "skills": "Java, Spring Boot, Kafka, PostgreSQL, microservices",
            "history": "Backend Engineer 3.5 years, Lead Engineer 2.2 years, payments systems",
            "behavior": "Good mentorship and ownership, weak AI stack exposure",
            "years": (5.5, 8.0),
        },
        {
            "skills": "Python, FastAPI, LlamaIndex, Pinecone, Airflow, Terraform, distributed systems",
            "history": "Founding Engineer 4.0 years, Staff Engineer 3.1 years, automation platform architect",
            "behavior": "Open source contributor, strong ownership from scratch, technical leadership",
            "years": (7.0, 11.0),
        },
    ]

    rows: List[Dict[str, Any]] = []
    for index in range(50):
        archetype = archetypes[index % len(archetypes)]
        years = round(random.uniform(*archetype["years"]), 1)
        if index in {7, 19, 31, 43}:
            history = "AI Engineer 0.8 years, Backend Engineer 0.7 years, Consultant 0.6 years, short stint pattern"
            behavior = "Frequent switches, job hopper risk, good interview projects"
        else:
            history = archetype["history"]
            behavior = archetype["behavior"]

        rows.append(
            {
                "candidate_id": f"CAND-{index + 1:03d}",
                "full_name": f"{first_names[index % len(first_names)]} {last_names[(index * 3) % len(last_names)]}",
                "skills": archetype["skills"],
                "years_of_experience": years,
                "resume_text": history,
                "platform_activity": behavior,
            }
        )

    return pd.DataFrame(rows)


def create_gemini_client() -> Tuple[Any, RuntimeState]:
    state = RuntimeState()
    api_key = os.environ.get("GEMINI_API_KEY")
    if genai is None:
        state.disable_gemini("google-genai is not installed; using deterministic local fallback mode.")
        return None, state
    if not api_key:
        state.disable_gemini("GEMINI_API_KEY is not set; using deterministic local fallback mode.")
        return None, state

    try:
        return genai.Client(api_key=api_key), state
    except Exception as exc:
        state.disable_gemini(f"Could not initialize Gemini client: {exc}")
        return None, state


def print_markdown_report(final_df: pd.DataFrame, output_path: Path, state: RuntimeState) -> None:
    top_five = final_df.head(5)
    print("\n# Top 5 Candidate Shortlist\n")
    print(f"Output CSV: `{output_path}`\n")
    if not state.gemini_enabled:
        print(f"> Fallback mode active: {state.fallback_reason}\n")

    for _, row in top_five.iterrows():
        print(f"## Rank {int(row['Final_Rank'])}: {row['Full_Name']} ({row['Candidate_ID']})")
        print(
            f"- Semantic Score: **{row['Semantic_Score']:.2f}** | "
            f"Experience Score: **{row['Experience_Score']:.2f}** | "
            f"Hybrid Score: **{row['Hybrid_Score']:.2f}**"
        )
        print(f"- Fit Analysis: {row['LLM_Fit_Analysis']}\n")


def run_pipeline(csv_path: Optional[str], output_path: str, jd_text: str) -> pd.DataFrame:
    client, state = create_gemini_client()
    analyzer = JobDescriptionAnalyzer(client, state)
    requirements = analyzer.analyze(jd_text)
    embedding_engine = EmbeddingEngine(client, state)
    evaluator = CandidateEvaluator(requirements, embedding_engine)

    raw_candidates = evaluator.load_candidates(csv_path)
    profiles = evaluator.normalize_candidates(raw_candidates)
    scored = evaluator.score(profiles, jd_text)
    top_15 = scored.head(15)

    reranker = GeminiReranker(client, state, requirements)
    refined_top_15 = reranker.rerank(top_15, jd_text)

    remainder = scored[~scored["Candidate_ID"].isin(refined_top_15["Candidate_ID"])].copy()
    if not remainder.empty:
        remainder["LLM_Fit_Analysis"] = (
            "Not included in the top-15 qualitative refinement batch. Local hybrid score remains available for triage."
        )
        start_rank = len(refined_top_15) + 1
        remainder = remainder.sort_values("Hybrid_Score", ascending=False).reset_index(drop=True)
        remainder["Final_Rank"] = range(start_rank, start_rank + len(remainder))

    final_df = pd.concat([refined_top_15, remainder], ignore_index=True)
    output_columns = [
        "Candidate_ID",
        "Full_Name",
        "Semantic_Score",
        "Experience_Score",
        "LLM_Fit_Analysis",
        "Final_Rank",
    ]
    final_output = final_df[output_columns].sort_values("Final_Rank").reset_index(drop=True)
    output_file = Path(output_path)
    final_output.to_csv(output_file, index=False)

    print_markdown_report(final_df.sort_values("Final_Rank"), output_file, state)
    return final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidates for an AI/backend role using hybrid semantic scoring.")
    parser.add_argument("--candidates", help="Optional path to candidate CSV. If omitted, 50 mock candidates are generated.")
    parser.add_argument("--output", default="ranked_candidates.csv", help="Output CSV path.")
    parser.add_argument("--jd", default=DEFAULT_JOB_DESCRIPTION, help="Raw job description text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.candidates, args.output, args.jd)


if __name__ == "__main__":
    main()
