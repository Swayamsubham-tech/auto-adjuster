# Week 3 Retrospective
* **Status:** Computer Vision pipeline is complete, containerized (Dockerfile), and integration tests are passing.
* **Handoff:** The pipeline outputs a strictly formatted Damage JSON. This is the fixed contract that Week 4's RAG system will consume.
* **Known Limitations:** The classifier currently predicts damage TYPE only, not PART. The JSON outputs `"part": "unknown"`. This gap will be deferred to policy-matching heuristics in Week 4.
