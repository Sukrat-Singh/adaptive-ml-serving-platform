# Adaptive Model Serving Platform with Drift Detection
An end-to-end Machine Learning system designed for production environments. This project focuses on operational reliability by integrating model serving, input validation, automated data logging, and drift monitoring.

---

 ## Project Goal
 This project does:
 - Packaging a model behind a FastAPI service for real-time inference.
 - Implementing automated drift detection to identify when live data shifts from training distributions.
 - Enabling reproducible workflows using Docker and CI/CD pipelines

## Core Workflow

1. Train a baseline model on a tabular dataset.
2. Package the model behind a REST API.
3. Log predictions and input features.
4. Compare incoming data against training distribution.
5. Raise an alert when drift is detected.
6. Trigger a retraining pipeline.
7. Evaluate the new model against the current one.
8. Deploy the new model only if it performs better.
9. Track all versions and metrics.

---

### ⚙️ Work in Progress ⚙️
