"""
RAG Verification Agent.

Owner: teammate.

Grounds detected/claimed accessibility features against the Indian
Harmonised Guidelines for Universal Accessibility, so the final report can
cite an actual clause rather than just asserting a score.

RULES FOR THIS FILE:
  1. No calls to the shared Neon DB in here. This agent reads from a LOCAL
     vector index built once from the guidelines PDF (see
     build_guideline_index.py, you'll write that too) — that index is a
     file that ships in the repo (or is rebuilt from the PDF on setup), it
     is NOT the same thing as the shared Postgres/pgvector database.
  2. Embeddings MUST use sentence-transformers/all-MiniLM-L6-v2 (384-dim)
     — see contracts.py for why. Any other model breaks the eventual
     merge into the shared pgvector column.
  3. Take RAGVerificationInput, return RAGVerificationOutput. Nothing else.
"""

from app.agents.contracts import (
    RAGVerificationInput,
    RAGVerificationOutput,
    GroundedExplanation,
)


def verify_against_guidelines(input: RAGVerificationInput) -> RAGVerificationOutput:
    """
    PHASE 1 (do this first):
    Stub — one placeholder explanation per claim, no real citations yet.
    Commit this so the function signature exists and is importable.

    PHASE 2 (the real work):
    1. One-time (separate script, build_guideline_index.py):
       - Parse the Harmonised Guidelines PDF (pdfplumber or similar)
       - Chunk it (a few paragraphs per chunk, keep clause numbers attached)
       - Embed each chunk with all-MiniLM-L6-v2
       - Save chunks + embeddings to a local index (a simple .npy/.json
         pair is fine for now — no need for a real vector DB engine on
         your side yet, that happens at integration time)
    2. In this function, for each claim:
       - Embed a short query like "wheelchair ramp entrance requirement"
       - Cosine-similarity search the local index for the top few chunks
       - Call the LLM (Gemini) with those chunks + the claim, ask it to
         write a short explanation grounded ONLY in the retrieved text
       - Package the result as a GroundedExplanation with citations
    """
    explanations = [
        GroundedExplanation(
            feature=claim.feature,
            explanation=f"[stub] No guideline retrieval wired up yet for '{claim.feature}'.",
            citations=[],
        )
        for claim in input.claims
    ]

    return RAGVerificationOutput(
        location_id=input.location_id,
        explanations=explanations,
    )


if __name__ == "__main__":
    # Quick manual test: `python -m app.agents.rag_verification`
    from app.agents.contracts import Claim

    test_input = RAGVerificationInput(
        location_id="loc_test",
        category="mall",
        claims=[Claim(feature="ramp", status="detected", confidence=0.82)],
    )
    result = verify_against_guidelines(test_input)
    print(result.model_dump_json(indent=2))
