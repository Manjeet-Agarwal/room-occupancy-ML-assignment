"""Streamlit front-end for the room occupancy classifiers.

The five pipelines are trained in notebooks/train_models.ipynb and saved in
model/. This app only loads them and scores uploaded data; it never calls fit.
"""

from pathlib import Path
from urllib.parse import quote

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix

from utils import FEATURE_COLUMNS, METRIC_NAMES, MODEL_FILES, TARGET, score_predictions

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

CLASSES = [0, 1, 2, 3]
CLASS_NAMES = ["0 people", "1 person", "2 people", "3 people"]
READ_AS = {0: "empty", 1: "1 person", 2: "2 people", 3: "3 people"}

# Palette: slate/teal for the interface, lamp amber only for occupants
# (light is the strongest signal in this data: an empty room is a dark room).
INK = "#1B2A33"
SLATE = "#5B6B76"
MIST = "#EEF3F5"
TEAL = "#1F6F8B"
LAMP = "#E3A23B"
LINE = "#D6DEE3"
BEST_CELL = "#D7E9EF"

METRIC_HELP = {
    "Accuracy": "Share of rows where the predicted headcount is exactly right.",
    "AUC": "Macro average of one-vs-rest ROC AUC, computed from predicted probabilities.",
    "Precision": "Macro average: each headcount class counts equally.",
    "Recall": "Macro average: each headcount class counts equally.",
    "F1": "Macro average of the per-class F1 scores.",
    "MCC": "Matthews correlation coefficient: 1 = perfect, 0 = no better than chance.",
}

def person_icon(fill, stroke="none", dash=""):
    """Small head-and-shoulders pictogram as a CSS data URI."""
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 28'>"
           f"<g fill='{fill}' stroke='{stroke}' stroke-width='1.3' {dash}>"
           "<circle cx='10' cy='6' r='5'/><path d='M1.5 27v-6.5a8.5 8.5 0 0 1 17 0V27z'/></g></svg>")
    return "data:image/svg+xml," + quote(svg, safe="")


PERSON_ON = person_icon(LAMP)
PERSON_OFF = person_icon("none", SLATE, "stroke-dasharray='2.4 2'")

PAGE_CSS = f"""
<style>
h1 {{ font-stretch: 87.5%; letter-spacing: -0.01em; }}
.lede {{ color: {SLATE}; font-size: 1.05rem; line-height: 1.5; max-width: 62ch; margin: -0.6rem 0 0.4rem; }}
[data-testid="stMetricValue"] {{ font-stretch: 87.5%; font-weight: 600; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ color: {SLATE}; }}

.hc {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid {LINE};
       border-radius: 0.4rem; overflow: hidden; background: #FFFFFF; margin: 0.2rem 0 0.6rem; }}
.hc-cell {{ padding: 0.9rem 1.1rem 1rem; border-left: 1px solid {LINE}; }}
.hc-cell:first-child {{ border-left: 0; }}
.hc-cell.empty {{ background: {MIST}; }}
.hc-icons {{ height: 30px; display: flex; gap: 3px; align-items: flex-end; }}
.hc-person {{ display: inline-block; width: 19px; height: 27px;
               background: url("{PERSON_ON}") no-repeat center / contain; }}
.hc-person.off {{ background-image: url("{PERSON_OFF}"); }}
.hc-name {{ margin-top: 0.55rem; color: {SLATE}; font-size: 0.9rem; }}
.hc-rate {{ color: {INK}; font-size: 1.9rem; font-weight: 600; font-stretch: 87.5%;
            line-height: 1.15; font-variant-numeric: tabular-nums; }}
.hc-detail {{ color: {SLATE}; font-size: 0.85rem; margin-top: 0.15rem; }}
.hc-miss {{ color: {INK}; font-size: 0.85rem; margin-top: 0.45rem; line-height: 1.35; }}
.hc-miss.none {{ color: {SLATE}; }}
@media (max-width: 640px) {{
  .hc {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .hc-cell:nth-child(3) {{ border-left: 0; }}
  .hc-cell:nth-child(n+3) {{ border-top: 1px solid {LINE}; }}
}}
</style>
"""

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


# --------------------------------------------------------------------------- visuals
CM_CMAP = LinearSegmentedColormap.from_list("mist_teal", ["#FFFFFF", MIST, TEAL])


def confusion_heatmap(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(cm, annot=True, fmt="d", cmap=CM_CMAP, cbar=False, linewidths=1,
                linecolor="white", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted occupants", color=SLATE)
    ax.set_ylabel("Actual occupants", color=SLATE)
    ax.tick_params(colors=SLATE, length=0)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    return fig


def headcount_strip(y_true, y_pred):
    """One panel, four cells: how often each real headcount was predicted correctly."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cells = []
    for i, cls in enumerate(CLASSES):
        total, hit = int(cm[i].sum()), int(cm[i, i])
        icons = ("<span class='hc-person'></span>" * cls) if cls else "<span class='hc-person off'></span>"
        name = "Empty room" if cls == 0 else CLASS_NAMES[i]
        if total == 0:
            rate, detail, miss = "–", "not in this file", ""
        else:
            share = hit / total
            rate = "100%" if hit == total else f"{share:.1%}"
            detail = f"{hit:,} of {total:,} rows right"
            misses = [f"{int(cm[i, j])} read as {READ_AS[c]}"
                      for j, c in enumerate(CLASSES) if j != i and cm[i, j] > 0]
            miss = ("<div class='hc-miss'>" + ", ".join(misses) + "</div>" if misses
                    else "<div class='hc-miss none'>No misses</div>")
        cells.append(
            f"<div class='hc-cell{' empty' if cls == 0 else ''}'>"
            f"<div class='hc-icons'>{icons}</div>"
            f"<div class='hc-name'>{name}</div>"
            f"<div class='hc-rate'>{rate}</div>"
            f"<div class='hc-detail'>{detail}</div>{miss}</div>"
        )
    return "<div class='hc'>" + "".join(cells) + "</div>"


def report_table(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    # "accuracy" is a single number, not a per-class row; it is already shown as a metric
    report.pop("accuracy", None)
    table = pd.DataFrame(report).T
    table["support"] = table["support"].astype(int)
    return table


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
st.html(PAGE_CSS)

st.title("How many people are in the room?")
st.html("<p class='lede'>Five classifiers estimate the headcount (0 to 3 people) from "
        "temperature, light, sound, CO2 and motion sensors. Upload sensor readings to see "
        "how well each one counts.</p>")

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
    st.subheader("1. Upload test data")
    uploaded = st.file_uploader("CSV with the 16 sensor columns and the target", type="csv")
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv",
                           help="The 2,026-row held-out test split used in the notebook.")

    st.subheader("2. Choose a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()), index=4)
    st.caption(model_settings(models[model_name]))

    with st.expander("Required columns"):
        st.code("\n".join(FEATURE_COLUMNS + [TARGET]), language=None)

# ---- nothing uploaded yet
if uploaded is None:
    st.info("Upload a CSV in the sidebar to score the models. "
            "No file at hand? Download the sample `test_data.csv` from the sidebar and upload it.")
    benchmark = load_benchmark()
    if benchmark is not None:
        st.subheader("Scores from the training notebook")
        st.caption("Held-out test set, 2,026 rows. Precision, Recall and F1 are macro averages. "
                   "Shaded cells mark the best value in each column.")
        st.dataframe(benchmark.style.format("{:.4f}").highlight_max(axis=0, color=BEST_CELL))
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
scores = {name: score_predictions(y, pred, proba, models[name].classes_)
          for name, (pred, proba) in outputs.items()}

tab_model, tab_compare, tab_data = st.tabs([model_name, "Compare all five", "Uploaded data"])

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
    st.markdown("**How each headcount was recognised**")
    st.html(headcount_strip(y, y_pred))
    st.markdown(
        f"{total} of {len(df):,} rows misclassified: {between} mix up an empty room with an "
        f"occupied one, {headcount} get the number of people wrong."
    )

    left, right = st.columns([1, 1.25])
    with left:
        fig = confusion_heatmap(y, y_pred, f"{model_name} confusion matrix")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = report_table(y, y_pred)
        st.dataframe(report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                          "f1-score": "{:.4f}", "support": "{:,}"}))

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
    st.dataframe(table.style.format(fmt).highlight_max(axis=0, color=BEST_CELL))
    st.caption("Shaded cells mark the best value in each column. Precision, Recall and F1 are "
               "macro averages; AUC is macro one-vs-rest.")

    best_mcc = table["MCC"].idxmax()
    st.markdown(f"Highest MCC on this file: **{best_mcc}** ({fmt(table.loc[best_mcc, 'MCC'])}).")

    st.markdown("**Misclassified rows by model**")
    errors = pd.DataFrame(
        [error_breakdown(y, pred)[1:] for pred, _ in outputs.values()],
        index=list(outputs.keys()),
        columns=["empty ↔ occupied", "wrong headcount"],
    )
    long = errors.rename_axis("Model").reset_index().melt(
        "Model", var_name="Error type", value_name="Rows")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("Model:N", sort=list(errors.index), title=None,
                    axis=alt.Axis(labelLimit=220, labelFontSize=13)),
            x=alt.X("Rows:Q", title="misclassified rows", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Error type:N", title=None,
                            scale=alt.Scale(domain=list(errors.columns), range=[SLATE, TEAL]),
                            legend=alt.Legend(orient="bottom")),
            order=alt.Order("Error type:N"),
            tooltip=["Model", "Error type", "Rows"],
        )
    )
    st.altair_chart(chart, height=280)

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
