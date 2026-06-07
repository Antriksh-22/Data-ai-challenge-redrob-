# Redrob Data & AI Challenge - Intelligent Candidate Ranking

This repository contains a reproducible candidate-ranking solution for the Redrob Senior AI Engineer / founding-team hiring challenge.

## Idea

The system replaces simple keyword filtering with a hybrid, explainable ranking engine. It reads the Redrob job description, streams candidate profiles, and scores each candidate using evidence from their profile, skills, career history, and Redrob behavioral signals.

The goal is to find candidates who are not only keyword matches, but strong practical fits for Redrob's role: production AI/ML depth, retrieval and ranking experience, Python/backend engineering ability, product-shipping mindset, mentorship, and stable tenure.

## How The Ranking Works

Each candidate is converted into a normalized evidence text using:

- profile headline and summary
- current title and industry
- skills, proficiency, endorsements, and duration
- career history and role descriptions
- Redrob platform signals such as GitHub activity, recruiter response rate, interview completion, open-to-work status, notice period, and relocation preference

The local scoring formula combines:

- 25% semantic JD-to-profile alignment using sparse hashed vectors
- 25% AI/backend skill depth
- 20% experience fit, with a preference for the JD's 5-9 year band
- 15% behavioral quality and production ownership signals
- 10% tenure stability, penalizing repeated short stints
- 5% location and relocation fit

The code also supports optional Gemini refinement through `GEMINI_API_KEY`, but no API key is stored in the repository. If the environment variable is absent or the API is unavailable, the system falls back to deterministic local scoring and still completes.

## Results

The full `candidates.jsonl` dataset was streamed and ranked locally. The generated submission contains exactly 100 candidates and passes the official challenge validator.

Top recommended candidates:

| Rank | Candidate ID | Score | Summary |
| ---: | --- | ---: | --- |
| 1 | CAND_0018499 | 1.0000 | Senior Machine Learning Engineer with RAG, embeddings, recommendation systems, Pinecone, and Weaviate. |
| 2 | CAND_0046525 | 0.9935 | Senior Machine Learning Engineer with LLMs, pgvector, LangChain, Machine Learning, and LlamaIndex. |
| 3 | CAND_0042029 | 0.9870 | Senior Data Scientist with RAG, embeddings, and Weaviate. |
| 4 | CAND_0092278 | 0.9805 | Senior NLP Engineer with LLMs, pgvector, Machine Learning, and Milvus. |
| 5 | CAND_0005260 | 0.9740 | Senior NLP Engineer with Python, RAG, embeddings, pgvector, and MLOps. |

## Files

- `india_challenge_ranker.py` - challenge-aware ranker for the Redrob JSON/JSONL dataset
- `candidate_ranker.py` - generic modular ranker matching the original architecture brief
- `redrob_submission.csv` - official validator-ready top-100 submission file
- `ranked_candidates.csv` - detailed ranked output with candidate names and score fields
- `recommended_candidates_report.pdf` - PDF report of recommended candidates
- `Redrob_Idea_Submission_Filled.pptx` - completed idea submission deck
- `make_submission_assets.py` - regenerates the PPT and PDF artifacts
- `requirements.txt` - optional dependencies for Gemini and generic data-science mode

## Run

Place the challenge package locally and run:

```bash
python india_challenge_ranker.py --challenge-output redrob_submission.csv --detailed-output ranked_candidates.csv
```

To enable Gemini reranking, set the key in your shell before running:

```bash
set GEMINI_API_KEY=your_key_here
```

Then validate:

```bash
python path/to/validate_submission.py redrob_submission.csv
```

Expected result:

```text
Submission is valid.
```
