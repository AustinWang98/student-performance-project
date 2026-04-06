# Machine Learning Final Project — Group 7

Unsupervised **anomaly detection** on student performance data using **Isolation Forest** (scikit-learn): full write-up and modeling live in the **notebooks**; **`train_simplified_demo_model.py`** is a compact training script (same core idea, less elaborate than the notebook pipeline) that produces **`iforest_model.pkl`** for the **Streamlit** demo site **`streamlit_model_demo.py`**.

## Project layout

```text
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore                # What Git should ignore locally
├── train_simplified_demo_model.py  # Simplified training pipeline for the web demo → writes iforest_model.pkl (full logic lives in notebooks)
├── streamlit_model_demo.py   # Streamlit web app to demonstrate the trained model (run from repo root)
├── notebooks/
│   ├── 01_report_group7_isolation_forest.ipynb   # Main report: theory, use cases, workflow, sklearn, examples
│   ├── 02_basic_example_synthetic.ipynb          # 2D synthetic data + parameter sensitivity
│   └── 03_real_world_student_application.ipynb   # Student data pipeline + sensitivity
├── docs/
│   ├── final_project_group7_merged.pdf
│   └── website/
│       └── student_anomaly_detection_website.pdf
└── student/
    ├── student-mat.csv       # Math course (;-separated)
    ├── student-por.csv       # Portuguese course (;-separated)
    ├── student.txt           # Attribute dictionary + overlap note (382 students in both)
    └── student-merge.R       # R: merge keys to reproduce dual-course students
```

**Working directory:** Run `train_simplified_demo_model.py`, `streamlit_model_demo.py`, and Jupyter with the **repository root** as the current directory so paths like `student/...` and `iforest_model.pkl` resolve correctly. Notebooks under `notebooks/` load data via `../student/...`.

## Quick start

```bash
cd "/path/to/Final Project"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_simplified_demo_model.py   # creates iforest_model.pkl for the Streamlit demo
streamlit run streamlit_model_demo.py
```

Open the notebooks in order (`01_` → `03_`) with Jupyter or VS Code.

## What to commit to Git

**Do commit**

- `README.md`, `requirements.txt`, `.gitignore`
- `train_simplified_demo_model.py`, `streamlit_model_demo.py`
- Everything under `notebooks/`, `student/`, and `docs/` (optional: skip large PDFs if you prefer a lighter clone)

**Do not commit** (already listed in `.gitignore`)

- `.ipynb_checkpoints/` — Jupyter auto-saves; large and regenerable
- `iforest_model.pkl` — binary artifact; teammates regenerate with `train_simplified_demo_model.py`
- `__pycache__/`, virtual environments (`.venv/`, `venv/`)
- OS junk (`.DS_Store`), editor folders, `*.save` backups

If you intentionally want the pickle in the repo (e.g. for a demo without training), remove `*.pkl` from `.gitignore` or use `git add -f iforest_model.pkl`.

## Method summary

- **Data:** Math + Portuguese student tables concatenated; **`G1` / `G2` dropped** to avoid grade leakage in the unsupervised setup; **`G3`** may remain for analysis context.
- **Demo training script:** A streamlined version of the notebook pipeline—`ColumnTransformer` (numeric `StandardScaler`, categorical `OneHotEncoder`) + `IsolationForest` (`n_estimators=200`, `contamination=0.08`, `random_state=42`). See `train_simplified_demo_model.py`. Deeper analysis, sensitivity, and interpretation are in `notebooks/`.
- **Web demo:** `streamlit_model_demo.py` is a small Streamlit site that loads `iforest_model.pkl` and `student/*.csv` to showcase scores, plots, and an interactive check.

---

**Remote:** [github.com/AustinWang98/student-performance-project](https://github.com/AustinWang98/student-performance-project)

*University of Chicago — Machine Learning — Final Project (Group 7).*
