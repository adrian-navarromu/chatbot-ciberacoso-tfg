# Results

Raw evaluation artefacts produced by the experiments — the empirical evidence
behind every design decision. **This directory holds outputs, not code:** the
code that generates them lives in [`../src/evaluation/`](../src/evaluation/) and
in the numbered [`../notebooks/`](../notebooks/).

## Contents

- **`emotion_results/`** — per-model emotion classifier evaluation
  (BETO, RoBERTuito, MarIA ± class weights): metrics, confusion matrices, reports.
- **`rag_experiments/`** — the 4 factorial RAG experiments (chunking, embeddings,
  vector store, hybrid search) as JSON + the corpus analysis CSV.
- **`generations/`** — SLM × prompt generations, LLM-as-a-judge scores, the
  human-validation sample and the inter-rater agreement report.
- **`prompt_engineering/`** — prompt variant A/B/C/D rubric results.
- **`slm_benchmark/`** — 5-model SLM benchmark rubric results.
- **`slm_results/`** — aggregated benchmark and conversational JSON results.
- **`crisis_evaluation.csv`** — full CrisisDetector confusion-matrix evaluation.

The plots derived from these results are in [`../docs/figures/`](../docs/figures/);
their browsable notebook write-ups are in [`../reports/`](../reports/).
