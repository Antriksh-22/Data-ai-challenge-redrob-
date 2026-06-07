from __future__ import annotations

import csv
import datetime as dt
import html
import re
import shutil
import textwrap
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Tuple


WORKSPACE = Path(__file__).resolve().parent
TEMPLATE_PPTX = Path(r"C:\Users\FireFly\Downloads\Idea Submission Template _ Redrob.pptx")
FILLED_PPTX = WORKSPACE / "Redrob_Idea_Submission_Filled.pptx"
RANKED_CSV = WORKSPACE / "redrob_submission.csv"
DETAILED_CSV = WORKSPACE / "ranked_candidates.csv"
RANKED_PDF = WORKSPACE / "recommended_candidates_report.pdf"


def read_top_candidates(limit: int = 15) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with RANKED_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def sanitize_xml_text(text: str) -> str:
    return html.escape(text, quote=False)


def extract_text_nodes(xml: str) -> List[str]:
    return [html.unescape(match) for match in re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.DOTALL)]


def replace_slide_text(xml: str, replacements: Dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        original = html.unescape(match.group(1))
        replacement = replacements.get(original)
        if replacement is None:
            return match.group(0)
        return f"<a:t>{sanitize_xml_text(replacement)}</a:t>"

    return re.sub(r"<a:t>(.*?)</a:t>", repl, xml, flags=re.DOTALL)


def build_slide_replacements() -> Dict[int, Dict[str, str]]:
    top = read_top_candidates(5)
    top_lines = "\n".join(
        f"#{row['rank']} {row['candidate_id']} - score {row['score']}: {row['reasoning']}"
        for row in top
    )
    today = dt.date.today().isoformat()
    repo_url = "https://github.com/Antriksh-22/Data-ai-challenge-redrob-"

    return {
        1: {
            "Team Name :": "Team Name: Antriksh-22",
            "Problem Statement :": "Problem Statement: Intelligent Candidate Discovery & Ranking for Redrob AI",
            "Team Leader Name :": "Team Leader Name: Antriksh",
        },
        2: {
            "Solution Overview": "Solution Overview",
            "What is your proposed solution?": (
                "A hybrid AI ranking engine that identifies the best Senior AI Engineer candidates for Redrob. "
                "It combines JD understanding, production-signal scoring, semantic profile matching, behavioral indicators, "
                "and explainable final ranking into one reproducible pipeline."
            ),
            "What differentiates your approach from traditional candidate matching systems?": (
                "Traditional filters over-rank keyword stuffing. Our approach scores evidence: AI/backend skill depth, "
                "years fit, production ownership, recruiter activity, GitHub signal, tenure stability, location/relocation, "
                "and role context. Every selected candidate receives a concise reason tied to observed data."
            ),
        },
        3: {
            "JD Understanding & Candidate Evaluatio": "JD Understanding & Candidate Evaluation",
            "n": "",
            "What are the key requirements extracted from the JD?": (
                "Extracted requirements: 5-9 years preferred, production ML systems, embeddings, retrieval, ranking, "
                "LLM re-ranking, strong Python/backend capability, scrappy product shipping, evaluation infrastructure, "
                "mentorship, and founding-team ownership."
            ),
            "Which candidate signals are most important for determining relevance? / How does your solution evaluate candidate fit beyond keyword matching?": (
                "Important signals: core AI skills, profile summary, career history, skill proficiency/duration, "
                "production language in role descriptions, average tenure, GitHub activity, recruiter response rate, "
                "interview completion, open-to-work status, and relocation/location suitability."
            ),
        },
        4: {
            "Ranking Methodology": "Ranking Methodology",
            "How does your system retrieve, score, and rank candidates?": (
                "The pipeline streams all 100,000 JSONL profiles, converts each profile into a normalized evidence text, "
                "computes sparse hashed semantic similarity to the JD, and keeps only the strongest candidates in memory."
            ),
            "What models, algorithms, or heuristics are used?": (
                "Algorithms: deterministic hashed vector similarity, weighted skill matching, 5-9 year experience fit, "
                "behavioral multiplier, tenure penalty for short stints, location/relocation bonus, and optional Gemini "
                "reranking when GEMINI_API_KEY is available."
            ),
            "How are multiple candidate signals combined into a final ranking?": (
                "Final local score = 25% semantic + 25% AI/backend skill depth + 20% experience + 15% behavioral quality "
                "+ 10% tenure + 5% location fit, with guardrail penalties for weak core skills, low experience, and job hopping."
            ),
        },
        5: {
            "Explainability & Data Validation": "Explainability & Data Validation",
            "How are ranking decisions explained?": (
                "Each row includes a human-readable reason summarizing title, years, strongest AI/backend skills, behavior score, "
                "and tenure score. This makes the ranking auditable by recruiters and challenge reviewers."
            ),
            "How do you prevent hallucinations or unsupported justifications?": (
                "Justifications are generated from structured candidate fields only: profile, skills, career history, and Redrob signals. "
                "The default mode is deterministic and does not invent facts."
            ),
            "How does your solution handle inconsistent, low-quality, or suspicious profiles?": (
                "Missing values fall back to neutral defaults. Suspicious patterns such as short tenures, weak production evidence, "
                "low AI/backend skill depth, or poor engagement reduce the score instead of crashing the pipeline."
            ),
        },
        6: {
            "End-to-End Workflow": "End-to-End Workflow",
            "What is the complete workflow from JD input to ranked candidate output?": (
                "1. Read Redrob JD from DOCX. 2. Stream candidates.jsonl. 3. Build evidence text per candidate. "
                "4. Score semantic, skills, experience, behavior, tenure, and location. 5. Keep the top pool. "
                "6. Optionally rerank top 15 with Gemini. 7. Export valid 100-row CSV and ranked PDF report."
            ),
        },
        7: {
            "System Architecture": (
                "System Architecture\n"
                "JobDescriptionAnalyzer: reads/parses the JD.\n"
                "EmbeddingEngine: deterministic sparse vectors, optional Gemini embeddings/rerank path.\n"
                "CandidateEvaluator: streaming scorer over JSON/JSONL candidate data.\n"
                "GeminiReranker: optional qualitative top-15 refinement with safe fallback.\n"
                "Deliverables: redrob_submission.csv, ranked_candidates.csv, recommended_candidates_report.pdf, PPT."
            )
        },
        8: {
            "Results & Performance": "Results & Performance",
            "What results or insights demonstrate ranking quality?": (
                f"Top results are senior AI/ML/NLP engineers with strong retrieval, RAG, vector DB, Python, MLOps, and stable tenure signals.\n{top_lines}"
            ),
            "How does your solution meet the challenge's runtime and compute constraints?": (
                "The full 100,000-candidate dataset was streamed locally in about 98 seconds on this machine. "
                "The scorer uses standard-library sparse vectors, so it avoids large memory usage and remains free-tier friendly."
            ),
        },
        9: {
            "Technologies Used": "Technologies Used",
            "What technologies, frameworks, and tools were used and why were they": (
                "Python 3, CSV/JSON streaming, DOCX XML extraction, sparse hashed vectors, deterministic scoring heuristics, "
                "optional modern Google GenAI SDK, and the official Redrob validator."
            ),
            "selected for this solution?": (
                "These choices keep the solution reproducible, explainable, fast, and submission-safe. "
                "No secret keys are stored in code; Gemini is enabled only through GEMINI_API_KEY."
            ),
        },
        10: {
            "Submission Assets": "Submission Assets",
            "Github video etc": (
                f"GitHub repository: {repo_url}\n"
                "Final submission CSV: redrob_submission.csv, validated with official validate_submission.py.\n"
                "Ranked recommendations PDF: recommended_candidates_report.pdf.\n"
                f"Generated on: {today}."
            ),
        },
    }


def fill_pptx() -> None:
    if not TEMPLATE_PPTX.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE_PPTX}")
    replacements = build_slide_replacements()
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(TEMPLATE_PPTX, "r") as archive:
            archive.extractall(tmp_dir)

        slides_dir = tmp_dir / "ppt" / "slides"
        for slide_num, mapping in replacements.items():
            slide_path = slides_dir / f"slide{slide_num}.xml"
            if not slide_path.exists():
                continue
            xml = slide_path.read_text(encoding="utf-8")
            xml = replace_slide_text(xml, mapping)
            slide_path.write_text(xml, encoding="utf-8")

        if FILLED_PPTX.exists():
            FILLED_PPTX.unlink()
        with zipfile.ZipFile(FILLED_PPTX, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in tmp_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(tmp_dir).as_posix())


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_lines(text: str, width: int = 95) -> List[str]:
    lines: List[str] = []
    for chunk in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(chunk, width=width) or [""])
    return lines


def build_pdf_pages(rows: List[Dict[str, str]]) -> List[List[str]]:
    pages: List[List[str]] = []
    current: List[str] = [
        "Redrob Recommended Candidates Report",
        f"Generated: {dt.date.today().isoformat()}",
        "Role: Senior AI Engineer - Founding Team",
        "",
        "Scoring summary: semantic JD match, AI/backend skill depth, experience fit, behavioral signals, tenure stability, and location fit.",
        "",
    ]
    for row in rows:
        block = [
            f"Rank {row['rank']} | {row['candidate_id']} | Score {row['score']}",
            f"Reasoning: {row['reasoning']}",
            "",
        ]
        wrapped: List[str] = []
        for line in block:
            wrapped.extend(wrap_lines(line))
        if len(current) + len(wrapped) > 44:
            pages.append(current)
            current = []
        current.extend(wrapped)
    if current:
        pages.append(current)
    return pages


def make_pdf() -> None:
    rows = read_top_candidates(100)
    pages = build_pdf_pages(rows)
    objects: List[str] = []

    def add_object(body: str) -> int:
        objects.append(body)
        return len(objects)

    font_obj = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_objs: List[int] = []
    for page in pages:
        text_ops = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in page:
            text_ops.append(f"({pdf_escape(line)}) Tj")
            text_ops.append("T*")
        text_ops.append("ET")
        stream = "\n".join(text_ops)
        content_obj = add_object(f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream")
        page_obj = add_object(
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>"
        )
        page_objs.append(page_obj)

    kids = " ".join(f"{obj} 0 R" for obj in page_objs)
    pages_obj = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_objs)} >>")
    catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")

    for obj_index in page_objs:
        objects[obj_index - 1] = objects[obj_index - 1].replace("/Parent 0 0 R", f"/Parent {pages_obj} 0 R")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n{body}\nendobj\n".encode("latin-1", errors="replace"))
    xref_at = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")
    )
    RANKED_PDF.write_bytes(output)


def main() -> None:
    fill_pptx()
    make_pdf()
    print(f"Wrote {FILLED_PPTX}")
    print(f"Wrote {RANKED_PDF}")


if __name__ == "__main__":
    main()
