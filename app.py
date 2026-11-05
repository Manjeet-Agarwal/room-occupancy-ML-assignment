"""Streamlit front-end for the room occupancy classifiers.

The five pipelines are trained in notebooks/train_models.ipynb and saved in
model/. This app only loads them and scores uploaded data; it never calls fit.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from utils import FEATURE_COLUMNS, METRIC_NAMES, MODEL_FILES, TARGET, score_predictions

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

CLASSES = [0, 1, 2, 3]
CLASS_NAMES = ["0 people", "1 person", "2 people", "3 people"]

METRIC_HELP = {
    "Accuracy": "Share of rows where the predicted headcount is exactly right.",
    "AUC": "Macro average of one-vs-rest ROC AUC, computed from predicted probabilities.",
    "Precision": "Macro average: each headcount class counts equally.",
    "Recall": "Macro average: each headcount class counts equally.",
    "F1": "Macro average of the per-class F1 scores.",
    "MCC": "Matthews correlation coefficient: 1 = perfect, 0 = no better than chance.",
}


# --------------------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    # The .pkl files are produced by this project's own notebook, so they are trusted.
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_data
def load_benchmark():
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path, index_col="Model") if path.exists() else None


def read_upload(uploaded_file):
    """Return (dataframe, error message)."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # malformed / empty / not a CSV
        return None, f"Could not read the file as CSV ({exc})."
    if df.empty:
        return None, "The file has a header but no data rows."
    return df, None


def validate(df):
    """Check columns and values. Returns (X, y, list of problems)."""
    problems = []
    required = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append("Missing column(s): " + ", ".join(f"`{c}`" for c in missing))
        return None, None, problems

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_cols = X.columns[X.isna().any()].tolist()
    if bad_cols:
        problems.append("Empty or non-numeric values in: " + ", ".join(f"`{c}`" for c in bad_cols))
    inf_cols = X.columns[np.isinf(X).any()].tolist()
    if inf_cols:
        problems.append("Infinite values in: " + ", ".join(f"`{c}`" for c in inf_cols))

    y = pd.to_numeric(df[TARGET], errors="coerce")
    unknown = sorted(set(y.dropna().unique()) - set(CLASSES))
    if y.isna().any():
        problems.append(f"`{TARGET}` has empty or non-numeric values.")
    elif unknown:
        problems.append(f"`{TARGET}` must be 0, 1, 2 or 3; found {unknown}.")

    return X, (y.astype(int) if not problems else y), problems


def predict_all(models, X):
    return {name: (m.predict(X), m.predict_proba(X)) for name, m in models.items()}


def fmt(value):
    return "n/a" if pd.isna(value) else f"{value:.4f}"


# --------------------------------------------------------------------------- charts
def confusion_heatmap(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1,
                linecolor="white", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted occupants")
    ax.set_ylabel("Actual occupants")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def report_table(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T
    table["support"] = table["support"].astype(int)
    return table.round(4)


def error_breakdown(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wrong = y_true != y_pred
    empty_vs_occupied = wrong & ((y_true == 0) | (y_pred == 0))
    return int(wrong.sum()), int(empty_vs_occupied.sum()), int((wrong & ~empty_vs_occupied).sum())


def model_settings(pipe):
    clf = pipe.named_steps["clf"]
    keys = {
        "LogisticRegression": ["max_iter"],
        "DecisionTreeClassifier": ["max_depth"],
        "KNeighborsClassifier": ["n_neighbors"],
        "GaussianNB": ["var_smoothing"],
        "SVC": ["kernel", "C", "gamma"],
    }.get(type(clf).__name__, [])
    params = clf.get_params()
    shown = ", ".join(f"{k}={params[k]}" for k in keys)
    return f"StandardScaler → {type(clf).__name__}({shown})"


# --------------------------------------------------------------------------- page
st.set_page_config(page_title="Room Headcount Estimator", page_icon="🏢", layout="wide")

st.title("How many people are in the room?")
st.caption("Estimating room occupancy (0–3 people) from cheap, non-intrusive sensors "
           "with five scikit-learn classifiers.")

with st.expander("About this project", expanded=False):
    st.markdown(
        """
Counting people with cameras is accurate but intrusive. This project checks how well
**environmental sensors** alone can estimate the headcount in a 6 m × 4.6 m room:

| Sensor | Columns | What it reacts to |
|---|---|---|
| Temperature (4 nodes) | `S1_Temp` … `S4_Temp` | body heat, slowly |
| Light (4 nodes) | `S1_Light` … `S4_Light` | desk lamps switched on |
| Sound (4 nodes) | `S1_Sound` … `S4_Sound` | talking, movement |
| CO2 (1 node) | `S5_CO2`, `S5_CO2_Slope` | breathing; slope = rising or falling |
| Motion (2 PIR nodes) | `S6_PIR`, `S7_PIR` | movement under the ceiling sensor |

Data: *Room Occupancy Estimation*, UCI Machine Learning Repository.
The models were trained in `notebooks/train_models.ipynb` on 80 % of the data.
This app **only loads the saved models** and scores the file you upload.
        """
    )

models = load_models()

# ---- sidebar controls
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a CSV with the 16 sensor columns and the target",
                                type="csv")
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv",
                           help="The 2,026-row held-out test split used in the notebook.")

    st.header("2 · Model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()), index=4)
    st.caption(model_settings(models[model_name]))

    with st.expander("Required columns"):
        st.code("\n".join(FEATURE_COLUMNS + [TARGET]), language=None)

# ---- nothing uploaded yet
if uploaded is None:
    st.info("Upload a CSV in the sidebar to evaluate the models. "
            "Not sure what to upload? Download the sample `test_data.csv` from the sidebar.")
    benchmark = load_benchmark()
    if benchmark is not None:
        st.subheader("Scores recorded in the training notebook")
        st.caption("Held-out test set, 2,026 rows. Precision, Recall and F1 are macro averages.")
        st.dataframe(benchmark.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d5"))
    st.stop()

# ---- read + validate
df, read_error = read_upload(uploaded)
if read_error:
    st.error(read_error)
    st.stop()

X, y, problems = validate(df)
if problems:
    st.error("This file can't be scored yet:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()

extra = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET]]
if extra:
    st.caption("Ignored extra column(s): " + ", ".join(extra))

# ---- predict with every model (no fitting here)
try:
    outputs = predict_all(models, X)
except Exception:
    st.error("The models could not score this file. Check that it has the same format as test_data.csv.")
    st.stop()
scores ={name: score_predictions(y, pred, proba, models[name].classes_)
          for name, (pred, proba) in outputs.items()}

tab_model, tab_compare, tab_data = st.tabs(
    [f"{model_name}", "Compare all five", "Uploaded data"])

# ---- tab 1: selected model
with tab_model:
    y_pred, y_proba = outputs[model_name]
    s = scores[model_name]
    st.subheader(f"{model_name} on {len(df):,} uploaded rows")

    cols = st.columns(6)
    for col, metric in zip(cols, METRIC_NAMES):
        col.metric(metric, fmt(s[metric]), help=METRIC_HELP[metric], border=True)
    if pd.isna(s["AUC"]):
        st.caption("AUC is n/a because not every class (0–3) appears in the uploaded target column.")

    total, between, headcount = error_breakdown(y, y_pred)
    st.markdown(
        f"**{total}** of {len(df):,} rows misclassified — "
        f"**{between}** confuse an empty room with an occupied one, "
        f"**{headcount}** get the number of people wrong."
    )

    left, right = st.columns([1, 1.25])
    with left:
        fig = confusion_heatmap(y, y_pred, f"Confusion matrix — {model_name}")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        st.dataframe(report_table(y, y_pred))

    result = df.copy()
    result["predicted_count"] = y_pred
    result["correct"] = result["predicted_count"] == y.to_numpy()
    for i, cls in enumerate(models[model_name].classes_):
        result[f"prob_{cls}_people"] = y_proba[:, i].round(4)
    st.download_button(
        f"Download {model_name} predictions (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---- tab 2: side-by-side comparison
with tab_compare:
    st.subheader("All five models on the uploaded data")
    table = pd.DataFrame(scores).T[METRIC_NAMES]
    table.index.name = "Model"
    st.dataframe(table.style.format(fmt).highlight_max(axis=0, color="#cfe8d5"))
    st.caption("Green = best value in each column. Precision, Recall and F1 are macro averages; "
               "AUC is macro one-vs-rest.")

    best_mcc = table["MCC"].idxmax()
    st.markdown(f"Highest MCC on this file: **{best_mcc}** ({fmt(table.loc[best_mcc, 'MCC'])}).")

    st.markdown("**Macro-F1 and MCC by model**")
    st.bar_chart(table[["F1", "MCC"]], stack=False, horizontal=True,
                 y_label="", x_label="score", height=300)

    with st.expander("Confusion matrices for every model"):
        grid = st.columns(3)
        for i, (name, (pred, _)) in enumerate(outputs.items()):
            with grid[i % 3]:
                fig = confusion_heatmap(y, pred, name)
                st.pyplot(fig)
                plt.close(fig)

# ---- tab 3: the data itself
with tab_data:
    st.subheader("Preview")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))
    counts = y.value_counts().reindex(CLASSES, fill_value=0)
    st.markdown("**Actual headcount in this file**")
    st.dataframe(pd.DataFrame({
        "occupants": CLASSES,
        "rows": counts.values,
        "share": (counts / counts.sum()).round(4).values,
    }), hide_index=True)
