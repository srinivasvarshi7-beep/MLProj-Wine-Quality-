# Wine Quality Prediction – Streamlit Website

Three pages: **Prediction**, **Data Exploration**, **Model Information**.
Final model: **Tuned XGBoost** (multiclass, quality 3–8) trained on `WineQT.csv`.

## Project structure
```
wine_quality_app/
├── app.py                  <- the website (run this)
├── utils.py                <- loading + prediction helpers
├── train_model.py          <- recreates the model files (already run for you)
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml  <- dark theme + red accent
├── data/WineQT.csv
├── models/
│   ├── wine_quality_model.json    <- XGBoost model (native JSON format)
│   ├── model_metadata.json        <- feature order, label mapping, test metrics, feature importance
│   └── notebook_results.json      <- comparison table copied from your notebook output
└── notebook/Wine_Quality_XGBoost_LightGBM_Project.ipynb
```

## Important facts about this project
- Your notebook saved model files, but they were not uploaded, so the model was **retrained** with your notebook's exact
  steps (same split, `random_state=42`, label encoding, best parameters). It reproduces your notebook's test results exactly
  (Accuracy 69.87%, Weighted F1 68.32%, Macro F1 41.12%).
- The 11 features (in order): fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
  total sulfur dioxide, density, pH, sulphates, alcohol. The `Id` column is not used. No scaler is needed (tree models).
- Output is the exact quality score (3–8). Labels were encoded 0–5 for XGBoost; the app converts back using `model_metadata.json`.
- LightGBM was trained in the notebook but only its metrics are shown (it was not the final model), so `lightgbm` is not required.
- Random Forest is **not** in your notebook, so it does not appear on the site.
- Tested with Python 3.12, xgboost 3.4.1, scikit-learn 1.8.0, streamlit 1.65. Python 3.10–3.12 recommended.

## Run locally on Windows (VS Code)
1. Install Python 3.10–3.12 from python.org (tick **Add Python to PATH**). Install VS Code.
2. Extract the ZIP, then in VS Code: **File → Open Folder → wine_quality_app**.
3. Open a terminal (**Terminal → New Terminal**) and run:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.
4. The site opens at http://localhost:8501. Stop it with `Ctrl+C`.

To rebuild the model files (optional): `python train_model.py`.

## Upload to GitHub
1. Create a free account at github.com → **New repository** (e.g. `wine-quality-app`, public).
2. In the VS Code terminal:
```
git init
git add .
git commit -m "Wine quality app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-app.git
git push -u origin main
```
Commit **everything** in the folder: `app.py`, `utils.py`, `train_model.py`, `requirements.txt`, `.streamlit/`, `data/`,
`models/` (all three files) and `notebook/`. They are small (about 2 MB in total). Do not commit `venv/` (`.gitignore` skips it).
The model is a JSON file, not a pickle, so there is no unsafe-loading risk.

## Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and sign in with GitHub.
2. **Create app** → choose your repository, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose Python 3.11 or 3.12, then **Deploy**.

## Troubleshooting
- **"Model file not found"** – the `models/` folder must sit next to `app.py`. Run `python train_model.py` to recreate it.
- **`ModuleNotFoundError`** – the virtual environment is not active or `pip install -r requirements.txt` was not run.
- **XGBoost load/predict error after upgrading** – run `python train_model.py` to regenerate the model with your installed version.
- **Deployment build fails** – check that `requirements.txt` is in the repository root and Python version is 3.10–3.12.
- **`streamlit` not recognised** – activate the venv first (`venv\Scripts\activate`).

*The predictions are statistical estimates for learning purposes, not a professional wine-tasting assessment.*
