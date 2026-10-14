"""
Wine Quality Prediction - Streamlit website
Pages: Prediction | Data Exploration | Model Information
Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import utils

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

# ---- small amount of CSS to polish the look (dark theme itself is in .streamlit/config.toml) ----
st.markdown("""
<style>
h1, h2, h3 { color: #FFFFFF; }
div.stButton > button {
    background-color: #E5383B; color: white; border: none; font-weight: 600;
    padding: 0.6rem 1.6rem; border-radius: 6px;
}
div.stButton > button:hover { background-color: #BA181B; color: white; }
.result-card { background:#2A2A32; border-left:6px solid #E5383B; border-radius:8px; padding:1.2rem 1.5rem; }
.result-card .score { font-size:3rem; font-weight:700; color:#FFFFFF; line-height:1.1; }
.result-card .label { color:#BBBBC4; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

# ---- load everything once; show clear errors if files are missing ----
try:
    df = utils.load_dataset()
    meta = utils.load_json(str(utils.META_PATH))
    nb = utils.load_json(str(utils.NOTEBOOK_PATH))
    model = utils.load_model()
except Exception as e:  # noqa
    st.error(f"Could not load the project files: {e}")
    st.info("Make sure the `data/` and `models/` folders are next to app.py. "
            "If model files are missing, run `python train_model.py` in the project folder.")
    st.stop()

FEATURES = meta["features"]

# ---- sidebar navigation ----
st.sidebar.title("🍷 Wine Quality")
page = st.sidebar.radio("Navigation", ["Prediction", "Data Exploration", "Model Information"])
st.sidebar.markdown("---")
st.sidebar.caption("Final model: Tuned XGBoost\n\nDataset: WineQT.csv (1,143 wines)\n\n"
                   "Predictions are for learning purposes only.")


# =====================================================================================
# PAGE 1 - PREDICTION
# =====================================================================================
def prediction_page():
    st.title("Wine Quality Prediction")
    st.write("Set the physicochemical properties of a wine below. The trained **Tuned XGBoost** model "
             "will predict its quality score (3–8) as it appears in the WineQT dataset.")

    st.subheader("Wine properties")
    values = {}
    left, right = st.columns(2)
    for i, feat in enumerate(FEATURES):
        s = meta["feature_stats"][feat]
        lo, hi, med = s["min"], s["max"], s["median"]
        # step size: 0.001 for tiny ranges, 0.01 for medium, 0.1 / 1.0 for large
        span = hi - lo
        step = 0.001 if span < 1.5 else 0.01 if span < 10 else 0.1 if span < 40 else 1.0
        col = left if i % 2 == 0 else right
        with col:
            values[feat] = st.slider(feat.title(), min_value=float(lo), max_value=float(hi),
                                     value=float(med), step=step,
                                     help=f"Dataset range: {lo:g} – {hi:g}  |  default = dataset median")

    st.write("")
    if st.button("Predict Wine Quality"):
        # Validation: warn about values in the extreme 1% tails (the model has seen very few such wines)
        extreme = [f for f in FEATURES
                   if values[f] < meta["feature_stats"][f]["p01"] or values[f] > meta["feature_stats"][f]["p99"]]
        try:
            score, probs = utils.predict_quality(model, meta, values)
        except Exception as e:  # noqa
            st.error(f"Prediction failed: {e}")
            return

        st.markdown(
            f"""<div class="result-card"><div class="label">Predicted quality score</div>
            <div class="score">{score}</div>
            <div class="label">Model confidence for this score: {probs[score]*100:.1f}%</div></div>""",
            unsafe_allow_html=True)
        if extreme:
            st.warning("These inputs are in the extreme tails of the training data, so the prediction is less "
                       "reliable: " + ", ".join(extreme))

        st.subheader("Probability for each quality score")
        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.bar([str(c) for c in probs], [p * 100 for p in probs.values()],
                      color=[utils.RED if c == score else "#6C6C78" for c in probs])
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        for b, p in zip(bars, probs.values()):
            ax.text(b.get_x() + b.get_width() / 2, p * 100 + 1.5, f"{p*100:.1f}%", ha="center", color=utils.TEXT, fontsize=9)
        utils.style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "These are the model's raw probabilities and may not be well calibrated. "
            "The result is a statistical estimate from 1,143 wines and is **not** a professional wine-tasting "
            "assessment. In the notebook's test set the model never correctly identified quality 3 or 4 "
            "(only 1 and 7 test examples), so be cautious about low scores.")


# =====================================================================================
# PAGE 2 - DATA EXPLORATION
# =====================================================================================
def exploration_page():
    st.title("Data Exploration")
    st.write("All tables and charts below are computed live from the supplied `WineQT.csv`.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))
    st.caption("The `Id` column is only an identifier, so the notebook drops it before modelling. "
               "The model therefore uses the other 11 columns as features and `quality` as the target.")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Columns and data types")
        info = pd.DataFrame({"Data type": df.dtypes.astype(str), "Missing": df.isna().sum()})
        st.dataframe(info, width="stretch")
    with right:
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe().T.round(3), width="stretch")

    st.subheader("Quality distribution (target variable)")
    counts = df["quality"].value_counts().sort_index()
    left, right = st.columns([2, 1])
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(counts.index.astype(str), counts.values, color=utils.RED)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", color=utils.TEXT)
        ax.set_xlabel("Wine quality")
        ax.set_ylabel("Number of wines")
        ax.set_title("Wine Quality Distribution")
        utils.style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        dist = pd.DataFrame({"Count": counts, "Share (%)": (counts / counts.sum() * 100).round(2)})
        dist.index.name = "Quality"
        st.dataframe(dist, width="stretch")
        st.caption("The classes are strongly imbalanced: qualities 5 and 6 make up "
                   f"{(counts[5] + counts[6]) / counts.sum() * 100:.1f}% of the data, while 3 and 8 are very rare.")

    st.subheader("Correlation heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap of Wine Features")
    utils.style_chart(fig)
    # colour-bar and tick labels need light text on the dark background
    ax.tick_params(colors=utils.TEXT)
    fig.axes[-1].tick_params(colors=utils.TEXT)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation of each feature with quality")
    cq = corr["quality"].drop(["quality", "Id"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(cq.index, cq.values, color=[utils.RED if v > 0 else "#6C8EBF" for v in cq.values])
    ax.set_xlabel("Correlation with quality")
    utils.style_chart(fig, ax)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        f"**Key findings (computed from the data):** alcohol has the strongest positive correlation with quality "
        f"(**{corr.loc['alcohol','quality']:.2f}**), followed by sulphates ({corr.loc['sulphates','quality']:.2f}) and "
        f"citric acid ({corr.loc['citric acid','quality']:.2f}). Volatile acidity has the strongest negative "
        f"correlation (**{corr.loc['volatile acidity','quality']:.2f}**). Correlation does not prove causation.")

    st.subheader("Feature vs quality")
    feat = st.selectbox("Choose a feature", FEATURES, index=FEATURES.index("alcohol"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="quality", y=feat, color=utils.RED, ax=ax,
                order=sorted(df["quality"].unique()),
                boxprops=dict(edgecolor="white"), whiskerprops=dict(color="white"),
                capprops=dict(color="white"), medianprops=dict(color="white"),
                flierprops=dict(markeredgecolor="#BBBBC4"))
    ax.set_xlabel("Wine quality")
    ax.set_ylabel(feat.title())
    ax.set_title(f"{feat.title()} vs Wine Quality")
    utils.style_chart(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================================
# PAGE 3 - MODEL INFORMATION
# =====================================================================================
def model_page():
    st.title("Model Information")
    st.write("Results below come from your notebook. Two gradient-boosting algorithms were trained: "
             "**XGBoost** (baseline, class-weighted, tuned) and **LightGBM** (baseline, tuned). "
             "Random Forest was not part of this notebook, so it is not shown.")

    st.subheader("Model comparison (held-out test set, 229 wines)")
    comp = pd.DataFrame(nb["comparison"]).set_index("Model")
    st.dataframe(comp.style.format("{:.2f}").highlight_max(axis=0, color="#7B1E22"), width="stretch")
    st.caption("Values are percentages copied from the notebook output. Highlighted cell = best in that column. "
               "Tuned XGBoost is best on all five test metrics.")

    st.subheader("Final model: Tuned XGBoost")
    m = meta["test_metrics"]
    cols = st.columns(5)
    for col, (lab, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision (weighted)", "precision_weighted"),
                                      ("Recall (weighted)", "recall_weighted"), ("F1 (weighted)", "f1_weighted"),
                                      ("F1 (macro)", "f1_macro")]):
        col.metric(lab, f"{m[key]:.2f}%")
    st.markdown(
        "**Why it was selected:** the notebook selects Tuned XGBoost because it scored highest on accuracy, "
        "weighted F1 and macro F1 on the test set. Note that LightGBM Tuned had a slightly higher "
        "*cross-validation* macro F1 (0.367 vs 0.352) but lower *test* scores, so the choice rests on a small test "
        "set (229 rows); the differences between models are small and should not be over-interpreted.")

    st.subheader("Cross-validation vs test-set results")
    cv = nb["cross_validation"]
    st.table(pd.DataFrame({
        "Tuned model": ["XGBoost", "LightGBM"],
        "Cross-validation macro F1 (3-fold, training data)": [f"{cv['XGBoost Tuned']['best_cv_macro_f1']:.4f}",
                                                              f"{cv['LightGBM Tuned']['best_cv_macro_f1']:.4f}"],
        "Test macro F1 (held-out 20%)": ["0.4112", "0.3349"],
    }))
    st.caption("Cross-validation was used only to choose hyperparameters (on the training split). "
               "Test-set numbers come from data the model never saw during tuning.")

    left, right = st.columns(2)
    with left:
        st.subheader("Hyperparameters (XGBoost)")
        st.table(pd.DataFrame(meta["best_params"].items(), columns=["Parameter", "Value"]).astype(str))
        st.caption("Found with RandomizedSearchCV (20 random combinations, 3-fold CV, scored by macro F1). "
                   "Fixed settings: objective=multi:softmax, num_class=6, random_state=42.")
    with right:
        st.subheader("Feature importance")
        fi = pd.Series(meta["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(fi.index, fi.values, color=utils.RED)
        ax.set_xlabel("Importance")
        utils.style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (test set)")
        cm = np.array(meta["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax, cbar=False,
                    xticklabels=meta["classes"], yticklabels=meta["classes"])
        ax.set_xlabel("Predicted quality")
        ax.set_ylabel("Actual quality")
        utils.style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report (test set)")
        rep = pd.DataFrame(meta["classification_report"]).T
        rep["support"] = rep["support"].astype(int)
        st.dataframe(rep.round(2), width="stretch")
        st.caption("Quality 3 and 4 have precision/recall of 0: the model never predicted them correctly "
                   "(only 1 and 7 test examples). Accuracy alone hides this, which is why macro F1 is reported.")

    st.subheader("Training and tuning process")
    st.markdown(f"""
1. Loaded `WineQT.csv` (1,143 rows), checked missing values (none) and duplicates (none), and dropped the `Id` column.
2. Split into {meta['split']['train_rows']} training and {meta['split']['test_rows']} test rows (80/20, stratified, `random_state=42`).
3. Label-encoded quality (3–8) to classes 0–5, as XGBoost requires.
4. Trained baseline XGBoost and LightGBM, plus an XGBoost experiment with balanced class weights (it did **not** help).
5. Tuned both with `RandomizedSearchCV` (macro F1) and compared all models on the test set.
6. Selected Tuned XGBoost and saved it.
""")
    st.caption("Model files for this website were regenerated by `train_model.py` using the notebook's documented "
               "procedure and best parameters; the regenerated model reproduces the notebook's test metrics exactly. "
               f"Versions used: XGBoost {meta['versions']['xgboost']}, scikit-learn {meta['versions']['scikit-learn']}.")


if page == "Prediction":
    prediction_page()
elif page == "Data Exploration":
    exploration_page()
else:
    model_page()
