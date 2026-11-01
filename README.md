# EEM-ACC: Self-Reflective Agentic AI with Adaptive Concurrency Control

Code for the thesis paper *"Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control"* (Bina Nusantara University).

- **EEM** (Execution Experience Memory): failure parsing, verbal reflection, Zettelkasten-style vector store, top-k retrieval.
- **ACC** (Adaptive Concurrency Controller): AIMD concurrency window driven by HTTP 429/503 and latency vs. `L_target`.

Work runs in Google Colab; heavy outputs (checkpoints, logs, results, figures, datasets) go to Google Drive, never to this repo.

See [`HANDOFF.md`](HANDOFF.md) for current status, decisions, and how to run things.
