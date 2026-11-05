# Room Occupancy Estimation from Environmental Sensors

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) that estimate how many people (0–3) are in a room from temperature, light, sound, CO2 and motion sensors, plus a Streamlit app that evaluates the saved models on uploaded test data.

- **Live app:** _link to be added after deployment_
- **Notebook:** [`notebooks/train_models.ipynb`](notebooks/train_models.ipynb)

---

## a. Problem statement

Knowing how many people are in a room lets a building adjust lighting, ventilation and heating to actual use instead of a fixed schedule. Cameras could count people, but they are intrusive and raise privacy concerns. The goal of this project is to predict the exact headcount in a room (0, 1, 2 or 3 people) using only cheap, non-intrusive environmental sensors, and to compare five classification models on that multi-class task.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Room Occupancy Estimation |
| Source | UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation> (DOI [10.24432/C5P605](https://doi.org/10.24432/C5P605)), licence CC BY 4.0 |
| Original work | A. P. Singh, V. Jain, S. Chaudhari, F. A. Kraemer, S. Werner, V. Garg, "Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes", 2018 IEEE Globecom Workshops |
| Size | 10,129 rows × 19 columns (`Date`, `Time`, 16 sensor features, 1 target) |
| Task | Multi-class classification, 4 classes |
| Missing values | None |

**Collection set-up.** A 6 m × 4.6 m room with seven wireless sensor nodes that report every 30 seconds; no HVAC was running during collection. The UCI page describes the collection as four days; the file itself contains readings on 7 dates (22–26 Dec 2017 and 10–11 Jan 2018).

**Features (16)**

| Sensor | Columns | Unit / range in the data |
|---|---|---|
| Temperature (nodes S1–S4) | `S1_Temp` … `S4_Temp` | °C, 24.4–29.0 |
| Light (nodes S1–S4) | `S1_Light` … `S4_Light` | lux, 0–280; median 0 for all four |
| Sound (nodes S1–S4) | `S1_Sound` … `S4_Sound` | amplifier output in volts, 0.04–3.88 |
| CO2 (node S5) | `S5_CO2` | ppm, 345–1,270 |
| CO2 slope (node S5) | `S5_CO2_Slope` | slope of CO2 over a sliding window, −6.3 to 9.0 |
| Motion (PIR, nodes S6–S7) | `S6_PIR`, `S7_PIR` | 0/1 |

**Target — `Room_Occupancy_Count`**

| Occupants | Rows | Share |
|---|---|---|
| 0 | 8,228 | 81.23 % |
| 1 | 459 | 4.53 % |
| 2 | 748 | 7.38 % |
| 3 | 694 | 6.85 % |

The classes are strongly imbalanced: always predicting "empty" would already give 81 % accuracy. This is why the split is stratified and why Precision, Recall and F1 are macro-averaged and MCC is reported.

**What the exploration showed**

- The features are on very different scales (CO2 in the hundreds, sound mostly below 0.1, PIR 0/1), so every model is wrapped in a `StandardScaler` pipeline.
- Light separates empty from occupied almost perfectly (median 0 lux empty vs 120–149 lux occupied) but separates 1/2/3 people less well. CO2 rises with headcount (medians 355 → 440 → 685 → 870 ppm), but the 2- and 3-person ranges overlap.
- 951 empty-room rows still have CO2 above 500 ppm, and 91 % of them have a falling CO2 slope: the room has just been vacated and the air has not cleared yet.
- The four temperature sensors are highly correlated with each other (r = 0.70–0.95), which breaks the independence assumption of Naive Bayes. The light sensors are the features most correlated with the target (S1_Light r = 0.85).

**Pre-processing**

- **Dropped `Date` and `Time`.** They say *when* a reading was taken, not *what* the sensors measured. With data from one room over a few days, a model given the timestamp could learn the experiment's schedule instead of the sensor patterns, which would not carry over to other days or rooms.
- **Split:** `train_test_split(test_size=0.2, stratify=y, random_state=55)` → 8,103 training rows and 2,026 test rows; class shares in both sets match the full data to within 0.05 percentage points. The test split is saved as [`test_data.csv`](test_data.csv) (16 features + target, 17 columns).
- **Time-series caveat:** the rows are a time series sampled every 30 s, and a random split puts neighbouring, near-identical readings into both train and test. 352 of the 2,026 test rows (17.4 %) have an exact duplicate in the training set. The test scores below are therefore optimistic; testing on whole unseen days would be a harder and fairer check.

## c. GitHub repository link

<https://github.com/Manjeet-Agarwal/room-occupancy-ML-classifier>

```
├── app.py                    # Streamlit app (loads saved models, never retrains)
├── utils.py                  # column lists + metric function shared by notebook and app
├── requirements.txt
├── test_data.csv             # held-out test split (2,026 rows) to upload in the app
├── data/
│   └── Occupancy_Estimation.csv
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── metrics.csv           # comparison table produced by the notebook
├── notebooks/
│   └── train_models.ipynb    # exploration, training, tuning, evaluation, saving
└── .streamlit/config.toml    # app theme
```

## d. Models used

All five models are scikit-learn `Pipeline`s of `StandardScaler` → classifier, trained on the same 8,103 rows and evaluated on the same 2,026 held-out rows. Hyper-parameters were chosen with 5-fold cross-validation on the training set only, using macro-F1 as the selection score.

| Model | Final settings | How they were chosen |
|---|---|---|
| Logistic Regression | `max_iter=1000` (lbfgs) | converged in 59 iterations; 1000 is a safety margin |
| Decision Tree | `max_depth=10` | grid over 2, 3, 4, 5, 6, 8, 10, 15, None |
| KNN | `n_neighbors=3` | grid over 3, 5, 7, 11, 15, 21 |
| Naive Bayes | `GaussianNB()` | Gaussian variant because the features are real-valued measurements, not counts (Multinomial NB expects counts) |
| SVM | `kernel="rbf"`, `C=10`, `gamma="scale"`, `probability=True` | grid over C = 0.1, 1, 10, 100 |

### Comparison table (held-out test set, 2,026 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9956 | **0.9998** | **0.9853** | 0.9872 | 0.9862 | 0.9865 |
| Decision Tree | 0.9951 | 0.9918 | 0.9819 | 0.9855 | 0.9835 | 0.9850 |
| KNN | 0.9921 | 0.9976 | 0.9743 | 0.9789 | 0.9765 | 0.9760 |
| Naive Bayes | 0.9571 | 0.9925 | 0.8726 | 0.9007 | 0.8846 | 0.8742 |
| Support Vector Machine | **0.9961** | 0.9997 | **0.9853** | **0.9907** | **0.9880** | **0.9880** |

**Averaging:** Precision, Recall and F1 use **macro** averaging — the score is computed for each of the four classes and then averaged with equal weight. With 81 % of rows in the "empty" class, weighted averaging would mostly reflect that one easy class; macro averaging makes the rare 1-, 2- and 3-person classes count just as much. **AUC** is the macro average of **one-vs-rest** ROC AUCs computed from `predict_proba`. MCC uses the full 4×4 confusion matrix.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A linear boundary works very well here (F1 0.9862, MCC 0.9865) because the strongest signals change steadily with headcount: light jumps from 0 to 120–150 lux as soon as anyone is present, and CO2 climbs from 355 to 870 ppm (medians) as people are added. Only 9 of 2,026 test rows are wrong and 6 of them are wrong headcounts, mostly 3 people predicted as 2 (5 rows), where the CO2 and light ranges of the two classes overlap. Train and test accuracy are almost equal (0.9951 / 0.9956), so it does not overfit. Its smooth probabilities give the best AUC of all models (0.9998). |
| Decision Tree | `max_depth` controlled the fit clearly: depths 2–3 underfit (CV macro-F1 0.720 and 0.877), CV F1 peaked at depth 10 (0.9833), and deeper trees (15 or unlimited) reached 100 % training accuracy while CV F1 dropped slightly to 0.9828 — extra depth only memorised the training rows. With depth 10 the gap is small (train 0.9996, test 0.9951) and test F1 is 0.9835 with 10 errors. It has the lowest AUC (0.9918) despite the third-best accuracy, because 99.4 % of test rows get a probability of exactly 1.0 for one class, so there is little ranking information for the ROC curve. The scaler has no effect on a tree; it is kept only so all pipelines have the same structure. |
| KNN | Sensitive to k in one direction: CV macro-F1 fell steadily from 0.9708 (k = 3) to 0.9687 (5), 0.9633 (7), 0.9605 (11) and 0.9543 (21), while accuracy fell much less. Larger neighbourhoods let the dominant empty class and the neighbouring headcounts outvote the rare classes. On the test set (k = 3) it is fourth (F1 0.9765, MCC 0.9760) with 16 errors, 8 of which confuse empty and occupied — the most of the four strong models. Scaling mattered, but not as expected: without the scaler KNN scored higher in CV (0.9860 vs 0.9708), because raw distances are dominated by CO2 and light, the two most informative sensors, whereas standardising gives the noisier sound and temperature sensors equal weight. It is also the largest model file (≈1.1 MB) because it stores all 8,103 training rows. |
| Naive Bayes | Clearly the weakest: accuracy 0.9571 looks acceptable, but macro-F1 (0.8846) and MCC (0.8742) show the problem, with recall for 3 people only 0.7266 and 87 errors in total. The correlated temperature sensors did hurt it, but in a specific way: 36 empty-room test rows were predicted as 3 people; these are just-vacated rooms with CO2 still high (median 838 ppm) and falling, and four warm temperature readings treated as independent evidence outweigh the low light. In CV, dropping the redundant temperature sensors halved these errors (109 → 55 → 31) but increased 3-predicted-as-2 errors (99 → 152 → 174), leaving total errors unchanged (289 / 292 / 291). The independence assumption is a real weakness, but the temperatures also carry useful signal, so removing them does not fix the model. |
| Support Vector Machine | Best on accuracy (0.9961), macro recall (0.9907), macro F1 (0.9880) and MCC (0.9880), with the fewest errors (8) and the best recall on the hardest class, 3 people (0.9712). `C = 10` was chosen by CV; `C = 100` reached 100 % training accuracy but scored lower in CV. Compared with linear Logistic Regression, the RBF kernel gains only one test row (8 vs 9 errors), which shows the problem is close to linearly separable in the scaled sensor space. Training took close to 1 s versus about 0.1 s for Logistic Regression (`probability=True` adds an internal 5-fold calibration) and the model file is 94.5 KB vs 2.6 KB. At 8,000 rows that extra cost is negligible, so it is worth it here, but the gain is small. |
| Overall, Winner for your dataset? | **SVM (RBF kernel).** It leads on accuracy, macro recall, macro F1 and MCC, ties on macro precision, and makes the fewest mistakes overall and on the rare 3-person class, which is where the models differ most. Logistic Regression is a very close second (one more error, best AUC, much faster and a far smaller model) and would be the better choice if training speed or simplicity mattered. Two caveats: the margin between the two is a single test row, and the random split is optimistic for time-series data (17.4 % of test rows have an exact twin in the training set), so a day-based split would be the next check. Naive Bayes is clearly last because its independence assumption does not fit sensors that move together. |

## Streamlit app

The app (`app.py`) loads the five saved pipelines once (`@st.cache_resource`) and never calls `fit`.

- **Sidebar:** CSV upload, a download button for the sample `test_data.csv`, model dropdown showing the chosen model's settings, and the list of required columns.
- **Validation:** if any of the 16 feature columns or the target column is missing, a value is empty, non-numeric or infinite, or the target is not 0–3, the app shows an error naming the problem columns instead of crashing.
- **Selected model tab:** the six metrics; a headcount panel with one cell per class (empty room, 1, 2, 3 people) showing how many rows of that class were predicted correctly and what the misses were read as; a one-line summary of the mistakes (empty vs occupied vs wrong headcount); the confusion matrix heatmap; the classification report; and a download of the predictions with class probabilities.
- **Look:** custom theme in `.streamlit/config.toml` (Archivo typeface, slate/teal interface colours) and a small stylesheet in `app.py`. Amber is used only for the person pictograms, a nod to the light sensors being the strongest signal: an empty room is a dark room.
- **Compare all five tab:** the six metrics for every model on the uploaded file with the best value in each column highlighted, a chart of misclassified rows per model (empty ↔ occupied vs wrong headcount) and all five confusion matrices.
- **Uploaded data tab:** preview and class counts of the uploaded file.

Metric code lives in `utils.py` and is shared with the notebook, so the app reproduces the notebook's table exactly when `test_data.csv` is uploaded.

## Run locally

Python 3.10.12, scikit-learn 1.7.2.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain, also install `jupyter` and run `notebooks/train_models.ipynb` from the `notebooks/` folder; it rewrites `test_data.csv` and everything in `model/`.
