# Netflix Dataset — Experiments 12 and 13

Adapted from the supplied HR MLOps project to `netflix.csv`. Since Netflix has no attrition label, the classification target is `type` (Movie=0, TV Show=1). This is a content-type classification task, not a prediction of viewer preference or title success.

## Run on Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py
python train_baseline.py
python experiment_12/run_experiment12.py
python experiment_13/end_to_end_pipeline.py
```

Experiment 12 measures test metrics, performs statistical feature-drift checks against a simulated shifted batch, trains a candidate model, and promotes it only if its F1-score is not lower than baseline. Experiment 13 executes the complete workflow. Reports are saved in `outputs/`. The shifted batch is synthetic for lab demonstration, not a live feed.
