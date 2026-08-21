# Explainable Crop Recommendation System

This repository is a tabular-data crop recommendation prototype. It accepts seven
numeric measurements—`N`, `P`, `K`, `temperature`, `humidity`, `ph`, and
`rainfall`—and combines a saved classifier with Top-3 model probabilities, global
feature importance, per-prediction SHAP attribution, and a separate provisional
rule-based soil assessment in one Streamlit application.

The intended research position is an **integrated decision-support framework**, not
a claim that Random Forest, crop recommendation, fertilizer guidance, weather input,
or the user interface is novel. The application is for research and demonstration;
it is not a substitute for region-specific advice from a qualified agronomist.

## Current scope

- Manual entry of all seven numeric model features.
- Optional place search and current-weather retrieval through Open-Meteo.
- Location-assisted mapping of current temperature and relative humidity while
  retaining manual soil values and model rainfall.
- Reproducible dataset validation and an 80/20 stratified experiment split.
- Comparison of Logistic Regression, Decision Tree, K-Nearest Neighbors, Support
  Vector Machine, Gaussian Naive Bayes, and Random Forest on the same held-out split.
- Saved-model inference with one recommended crop and exactly three ranked class
  probabilities.
- Deterministic local SHAP attribution under **Why this crop?**, plus global Random
  Forest feature importance in a separate technical view.
- Soil-status reporting that remains independent of the crop prediction.
- A read-only **Model & Research Evaluation** dashboard backed by stored artifacts.
- A responsive, high-contrast two-mode Streamlit dashboard that performs inference
  only after the user selects **Predict Crop**.

Location assistance is a Version 2 application enhancement. It does not alter the
dataset, training split, saved classifier, encoder, or measured research results.
Manual Input remains the reproducible baseline path.

## Data provenance and license

The specified source is Atharva Ingle's
[Crop Recommendation Dataset on Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset),
dataset identifier `atharvaingle/crop-recommendation-dataset`. Kaggle's data card
describes it as augmented rainfall, climate, and fertilizer data for India and lists
the dataset license as **Apache 2.0**. That license statement applies to the upstream
dataset; it does not imply a license for this repository's code.

Use the source file named `Crop_recommendation.csv` without changing its values, and
place it at:

```text
data/Crop_recommendation.csv
```

The included provenance record, `data/dataset_metadata.json`, identifies Kaggle
version 1 and records SHA-256
`54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e` for the
current CSV. Recompute and compare the checksum whenever the source is downloaded
again.

The validation step expects 2,200 rows, the seven feature columns in the exact order
shown below, one `label` target column, 22 crop labels, no missing values, and no
duplicate rows. These are validation expectations, not permission to silently delete
or rewrite unexpected data.

| Order | Field | Role |
|---:|---|---|
| 1 | `N` | Soil nitrogen measurement |
| 2 | `P` | Soil phosphorus measurement |
| 3 | `K` | Soil potassium measurement |
| 4 | `temperature` | Temperature in °C |
| 5 | `humidity` | Relative humidity in % |
| 6 | `ph` | Soil pH |
| 7 | `rainfall` | Rainfall in mm |
| 8 | `label` | Crop class (target) |

Download and checksum-verify the public dataset through the repository helper with:

```bash
python src/download_data.py
```

The helper copies the original CSV out of its project-local KaggleHub cache, verifies
that source and destination checksums match, and refuses to overwrite a different
existing file unless `--overwrite` is explicitly supplied after inspection. The
training workflow does not depend on the cache path.

## Setup

Use Python 3.10 or newer. Create and activate an isolated Python environment, then
install the pinned project dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The weather integration uses the pinned `requests` dependency for bounded HTTP
requests. Research commands use pinned `shap` for experimental explanations and
`pytest` for the complete automated suite. The basic Open-Meteo geocoding and
forecast endpoints used here require no API key, paid subscription, or user account.

## Input modes

### Manual Input Mode

Manual Input is selected by default. Enter `N`, `P`, `K`, temperature (°C), relative
humidity (%), soil pH, and rainfall (mm), then select **Predict Crop**. This is the
baseline application path corresponding most closely to the original tabular
experiment. It makes no weather-service request.

### Location-Assisted Input Mode

Select **Location-Assisted Input**, enter a place, and select **Search Location**.
The app sends that place text to the
[Open-Meteo geocoding service](https://open-meteo.com/en/docs/geocoding-api), shows
several matches when available, and requires you to choose a specific result. It
then uses the selected result's provider-supplied latitude and longitude—not a
silently assumed city—to retrieve current conditions from the
[Open-Meteo forecast service](https://open-meteo.com/en/docs).

Select **Get Weather** to retrieve data or **Refresh Weather** to bypass the short
weather cache. Requests are never made on each keystroke and weather is not refreshed
on a timer. The app displays the selected place, administrative region, country,
coordinates, timezone, provider, provider timestamp, current temperature, current
relative humidity, and current precipitation. Temperature is requested in Celsius
and maps to model `temperature`; relative humidity is requested as a percentage and
maps to model `humidity`.

`N`, `P`, `K`, soil pH, and model rainfall remain manual in this mode. Nutrient and
pH values should come from measurements or a compatible soil-test report; the app
does not retrieve or sense them. Both modes pass the same seven fields to
`src.pipeline.run_pipeline(...)`, which loads the same saved model and encoder. A
location lookup or weather refresh never trains a model.

#### Rainfall compatibility policy

The Kaggle data card describes the training feature only as **rainfall in mm**; it
does not document a daily, hourly, monthly, seasonal, or other measurement period.
Open-Meteo current precipitation is therefore displayed as contextual information
only. It is not automatically mapped to the model's `rainfall` field, even though
both use millimetres. The user must enter and confirm model rainfall separately.
This avoids presenting current model-step precipitation as scientifically equivalent
to an underspecified training feature. Live weather demonstrations are not used to
recompute model accuracy or support research-performance claims.

### Input flows

```text
Manual Input
  seven manual fields → saved inference pipeline → recommendation and explanations

Location-Assisted Input
  place search → explicit location selection → current Open-Meteo retrieval
  Open-Meteo temperature + humidity ─┐
  manual N/P/K/pH + model rainfall ──┴→ same saved inference pipeline
  current precipitation → context display only (never automatic model rainfall)
```

## Reproduce the workflow

Run commands from the repository root and keep the stages separate:

```bash
# 0. Download only if the verified CSV is not already present
python src/download_data.py

# 1. Inspect and validate the unmodified dataset
python src/preprocessing.py

# 2. Train and evaluate the specified Random Forest on the fixed split
python src/train.py

# 3. Recreate reliability, cross-validation, leakage, and confusion artifacts
python src/evaluate.py

# 4. Compare all six classifiers on that same split
python src/compare_models.py

# 5. Generate global feature-importance artifacts from the saved model
python src/explain.py

# 6. Exercise saved-model inference from the command line
python src/predict.py 90 42 43 25 80 6.5 200

# 7. Run all automated tests
python -m unittest discover -s tests -p "test_*.py"

# 8. Launch the two-mode application
streamlit run app.py

# 9. Run deterministic tests through pytest
pytest

# 10. Run the complete paper-oriented research suite
python -m src.research.run_all
```

Training uses an 80/20 stratified train/test split with `random_state=42`. The
specified Random Forest starts with 100 trees, `random_state=42`, and may use all
available cores. Five-fold stratified cross-validation is performed on the training
partition only; the held-out test partition must not influence fitting, scaling, or
cross-validation. Algorithms that need feature scaling use a scikit-learn pipeline so
their scalers are fitted only on training data.

Do not retrain from `app.py`. The frontend calls `src.pipeline.run_pipeline(...)`,
which reuses saved artifacts and combines prediction, explanation, and soil
assessment outputs.

## Project architecture

```text
                         USER
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Manual Input             Location-Assisted
             │                           │
             │                    Open-Meteo
             │                           │
             │                 Temp + Humidity
             │                 + precipitation
             │                    as context
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                    Input Validation
                           │
                           ▼
                Baseline Random Forest
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
      Recommended       Top-3          Probability
          Crop           Crops
            │
            ▼
        Local SHAP
       "Why this crop?"
            │
            ▼
      Soil Assessment
            │
            ▼
     Streamlit Dashboard

Research Evaluation
      │
      ├── Top-K
      ├── Calibration
      ├── SHAP
      ├── Tuning
      ├── Ablation
      ├── Robustness
      ├── Model Disagreement
      └── Error Analysis
```

The application has four final layers: machine learning, intelligent decision
support, the real-time Streamlit application, and stored research evaluation.
`src/predict.py` is the only saved-model prediction implementation;
`src/explain.py` contains the shared global and local SHAP compatibility layer;
`src/app_support.py` contains read-only production metadata, Top-3 presentation, and
research-summary loading. Research plotting and experiments remain under
`src/research/`. The pipeline never uses provisional soil categories to alter model
probabilities or reorder crop predictions.

## Experimental outputs

Measured files are written under `results/` and should be preserved for later paper
preparation:

```text
confusion_matrix.png
confusion_matrix.csv
classification_report.txt
evaluation_results.json
model_comparison.csv
model_comparison.json
model_accuracy_comparison.png
global_feature_importance.png
global_feature_importance.csv
```

This README intentionally does not quote accuracy, cross-validation, error counts,
model rankings, or importance percentages unless the corresponding artifacts have
been created by an actual run. Treat the generated JSON/CSV files—not historical
example values—as the source of truth. If two models tie in accuracy, compare macro
F1 and the other measured metrics, and document the tie rather than claiming a
unique winner.

## Measured rebuild results

The checked-in artifacts were generated from the unmodified, checksum-verified CSV
using the documented split and dependency versions. The Random Forest correctly
classified 438 of 440 held-out rows: accuracy `0.9954545`, macro precision
`0.9956710`, macro recall `0.9954545`, and macro F1 `0.9954517`. The two observed
errors were `blackgram → maize` and `rice → jute`.

Five-fold stratified cross-validation on the 1,760-row training partition produced
fold accuracies `1.000000`, `0.997159`, `0.982955`, `0.991477`, and `0.997159`, with
mean `0.993750` and standard deviation `0.006067`. The exact seven-feature-vector
overlap count between training and held-out test partitions was zero.

| Model | Held-out accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 0.995455 | 0.995452 |
| Gaussian Naive Bayes | 0.995455 | 0.995443 |
| Support Vector Machine | 0.984091 | 0.984038 |
| Decision Tree | 0.979545 | 0.979423 |
| K-Nearest Neighbors | 0.979545 | 0.979283 |
| Logistic Regression | 0.972727 | 0.972464 |

Random Forest and Gaussian Naive Bayes tied in held-out accuracy. Random Forest was
selected because its measured macro F1 was slightly higher; this is not presented as
a unique accuracy win. For the documented functional input (`90, 42, 43, 25, 80,
6.5, 200`), the saved Random Forest returned Rice `54%`, Jute `45%`, and Watermelon
`1%`. These are raw `predict_proba()` outputs for that input, not calibrated
guarantees or claims about real-world field performance.

## Final production application decision

The baseline `models/random_forest_crop.joblib` remains the only production model.
Both Manual and Location-Assisted modes pass the same seven ordered features through
the same validation and prediction pathway. Location never selects, retrains, or
updates a model. The tuned Random Forest produced the same held-out accuracy and
macro F1 as the baseline but worse log loss, so it remains isolated under
`models/research/`. The sigmoid-calibrated model did not improve overall measured
probability quality and also remains experimental. The UI therefore calls the raw
output **Prediction Probability**; it does not describe that value as guaranteed
confidence, crop-success probability, or yield probability.

After a successful prediction, the app uses the same baseline model to calculate
predicted-class SHAP contributions. The farmer-facing **Why this crop?** section
shows up to five influential inputs and explicitly labels whether each supports or
opposes the model output. An expandable technical view displays all seven numeric
SHAP contributions and an in-memory chart. No live explanation is saved to disk.
SHAP failure is isolated: the recommendation and Top-3 probabilities remain usable
and the UI reports that detailed explanation is temporarily unavailable.

The **Model & Research Evaluation** view reads existing JSON/CSV/PNG artifacts only.
Opening it never starts model training, tuning, calibration, ablation, robustness,
or research SHAP execution. Optional missing research summaries or charts are
reported without disabling crop prediction.

## Research Evaluation

The selected reference system lacks integrated location-assisted live weather input
and explainable reasoning for the predicted crop; this implementation extends the
baseline with these capabilities while also evaluating model reliability. **These
features represent enhancements relative to the selected reference system.** They
are not claims that weather APIs, SHAP, Random Forest, or Top-K recommendation are
globally new in agricultural research.

All experiments reconstruct the same ordered seven-feature, 1,760/440 stratified
split with `random_state=42`. Model selection and calibration fitting use the
training partition only. Experimental files are isolated under `results/research/`
and `models/research/`; neither experimental model is used by `app.py`.

### Measured research results

- **Top-K:** Baseline Top-1 accuracy is `0.995455` (438/440). Both mistakes have the
  true crop at rank two, producing Top-2 and Top-3 accuracy of `1.000000`. This only
  shows model-ranked alternatives on the benchmark; it does not make those crops
  agronomically interchangeable.
- **Probability calibration:** Five-fold training-only sigmoid calibration produced
  held-out accuracy `0.993182`, macro F1 `0.993178`, log loss `0.083758`, and
  10-bin top-label ECE `0.070277`. The raw baseline values were `0.995455`,
  `0.995452`, `0.050389`, and `0.037568`. The declared decision rule therefore
  finds no probability-quality improvement, so production inference remains raw.
  The plot uses multiclass top-label reliability: each sample contributes its
  maximum probability and whether that top label is correct. This follows the
  cross-validated fitting principle in the
  [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
- **SHAP explainability:** With SHAP `0.52.0`, the verified output shape is
  `440 × 7 × 22`. Mean absolute SHAP ranks humidity, rainfall, and K first; its
  feature-rank correlation with RF impurity importance is `0.928571`. SHAP and
  impurity importance differ because one summarizes contribution magnitudes over
  evaluated samples/classes while the other summarizes fitted-tree split usage.
  The implementation handles the documented multiclass array change described by
  [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).
- **Hyperparameter tuning:** A 16-candidate, five-fold `RandomizedSearchCV` selected
  `n_estimators=200`, `max_depth=None`, `max_features="sqrt"`,
  `min_samples_split=10`, `min_samples_leaf=1`, and `class_weight="balanced"`.
  Tuned accuracy and macro F1 tie the baseline exactly (`0.995455` and `0.995452`),
  while tuned log loss is worse (`0.062855` versus `0.050389`). The tuned model does
  not replace production.
- **Feature ablation:** Removing rainfall caused the largest macro-F1 change
  (`-0.027615`); removing no feature improved macro F1. This is benchmark-model
  dependence, not evidence that rainfall causally determines crop suitability.
- **Controlled robustness:** Across 110 stratified held-out samples, each feature had
  440 non-zero perturbations. Humidity and N each caused six Top-1 flips (`0.013636`
  flip rate); humidity ranked most sensitive under the declared flip-rate then
  probability-change ordering. These perturbations are numerical sensitivity tests,
  not forecasts or field-validation evidence.
- **Inter-model disagreement:** Of 440 held-out samples, the six original classifier
  definitions agreed 6/6 on 417, 5/6 on 13, 4/6 on 8, and 3/6 on 2. Ten samples had
  at most 4/6 modal agreement. This is **inter-model disagreement**, not formal
  uncertainty.
- **Error analysis:** The baseline still has exactly two errors: blackgram → maize
  and rice → jute. Both true crops appear second in the RF ranking. The two errors
  have mean Top-1 probability `0.635` and mean top-two margin `0.350`, compared with
  `0.959361` and `0.931735` for 438 correct cases. Two cases are far too few for a
  general causal or reliability conclusion.

Random Forest impurity importance, global SHAP importance, and feature-ablation
performance are retained as three distinct forms of evidence. Their agreement or
disagreement must not be interpreted as proof of an agronomic mechanism.

### Reproduce individual experiments

```bash
python -m src.research.evaluate_topk
python -m src.research.calibrate_probabilities
python -m src.research.shap_explain
python -m src.research.tune_random_forest       # full 16-candidate search
python -m src.research.tune_random_forest --quick  # development only
python -m src.research.feature_ablation
python -m src.research.robustness_analysis
python -m src.research.model_disagreement
python -m src.research.error_analysis
python -m src.research.summarize_research
```

The measured narrative is in `results/research/findings.md`; consolidated model
metrics are in `research_summary.csv` and `research_summary.json`. Figures are
paper-ready research artifacts but are not yet final IEEE-layout figures.

## Deployment readiness and required artifacts

From a fresh clone, create a Python 3.10+ virtual environment, install
`requirements.txt`, and run `streamlit run app.py`. The application needs these
committed production artifacts and never downloads or retrains them automatically:

```text
models/random_forest_crop.joblib
models/label_encoder.joblib
```

`results/global_feature_importance.png` and all `results/research/` files are presentation
artifacts; the app handles optional research artifacts gracefully when absent. The
model and encoder are required for prediction, and startup shows a clear restoration
message if either is missing. Open-Meteo requires network access for Location Mode
but no API key or repository secret. Manual Mode remains independent of the weather
service. `.venv`, `.env`, Python/test caches, temporary SHAP caches, and common IDE
metadata are excluded by `.gitignore`.

## Interpretation and scientific contribution

The defensible contribution is the integration and evaluation of:

- a controlled comparison of six established tabular classifiers;
- Top-K recommendation using the selected model's real `predict_proba()` output;
- transparent display of the predicted-class probability;
- global model-level feature importance; and
- an explicitly separate soil nutrient assessment.

This is not evidence that the underlying dataset, Random Forest, Top-K classification,
or soil assessment is new. Software conveniences such as Streamlit presentation,
downloads, voice support, or multilingual text are also not scientific novelty.
Claims in a paper should be based only on saved experiment artifacts and should cite
the relevant prior work.

## Limitations and safety notes

- **Dataset scope:** The public dataset is commonly used and is described as derived
  from Indian agricultural context. High performance on its held-out rows would be
  dataset-specific; it does not establish performance on farms, seasons, sensors, or
  regions not represented in the data. This is a benchmark dataset only; no real
  farm-field validation has been performed, and high benchmark accuracy does not
  guarantee field performance.
- **Model probability:** Random Forest `predict_proba()` is shown as prediction
  probability, not calibrated real-world certainty, guaranteed confidence, crop
  success, or yield probability. The measured calibration experiment did not justify
  replacing the raw production model.
- **Global importance:** Impurity-based `feature_importances_` summarizes the fitted
  model globally. It is not causal and is not a faithful local explanation for a
  single recommendation.
- **Local SHAP:** Predicted-class SHAP contributions describe behavior of this fitted
  model for one input. They do not establish agricultural causality, soil quality,
  fertilizer need, or guaranteed suitability.
- **Soil rules:** Until every threshold is backed by a documented, credible, and
  region-appropriate agricultural source, the application labels soil statuses as
  provisional or **Needs verification**. It does not issue specific fertilizer
  prescriptions or quantities from unverified rules. Interpretation depends on
  units, sampling and testing methods, region, and crop context.
- **Feature semantics:** The dataset's nutrient fields are described as ratios, but
  the data card does not establish a field sampling protocol or universal unit basis.
  N/P/K inputs require external soil measurements and must be made compatible with
  the training data before real-world use; the application does not invent units.
- **Weather Version 2:** Location mode retrieves current temperature, humidity, and
  precipitation context from Open-Meteo while retaining manual `N`, `P`, `K`, pH,
  and model rainfall. Current precipitation is not substituted for dataset rainfall.
  Live weather is time- and provider-dependent, may be missing, and does not validate
  model performance on that place. Network or geocoding failure does not affect the
  independent Manual Input mode.
- **Research reliability studies:** Robustness perturbations are synthetic numerical
  experiments, and inter-model disagreement is not formal uncertainty. Neither is a
  substitute for external validation.
- **Decision-support scope:** Top-3 crops are model-ranked alternatives, not proof that
  crops are interchangeable. The recommendation is decision support, not a guarantee.

## Weather errors, privacy, and troubleshooting

Place search and current conditions require internet access to Open-Meteo. Requests
have a finite timeout and the UI reports unknown locations, timeouts, network/HTTP
failures, malformed responses, and missing required measurements without inserting
fake defaults. If retrieval fails:

1. Check the network connection and try **Search Location** or **Refresh Weather**
   later.
2. Qualify an ambiguous place with a region or country and choose the intended match.
3. Continue in **Manual Input** mode; it does not depend on Open-Meteo.

For development without permanent external-network dependence, the automated suite
mocks geocoding and weather responses, including failure paths. User-entered place
text and selected coordinates are sent to Open-Meteo for lookup and conditions. The
app does not request browser GPS permission, claim GPS tracking, or retain a location
history. Consult Open-Meteo's service and privacy terms before public deployment.

## Version 2 change log

- Added modular Open-Meteo location search and current-weather retrieval with bounded
  requests, structured validation, explicit provider metadata, and short caching.
- Added optional Location-Assisted Input while retaining Manual Input as the default
  validated baseline path.
- Added safe mapping for Celsius temperature and percentage relative humidity.
- Kept current precipitation separate from manually confirmed model rainfall because
  the training feature's temporal semantics are undocumented.
- Added mocked provider/error tests and UI mapping regression tests without modifying
  the dataset, saved models, split, or measured experimental artifacts.

Open-Meteo integration is a software/application enhancement, not a claim of research
novelty, automatic soil sensing, GPS tracking, or real-world field validation.

## Suggested functional check

After the model and explanation artifacts have been generated, enter:

```text
N=90, P=42, K=43, temperature=25, humidity=80, ph=6.5, rainfall=200
```

Select **Predict Crop** and verify that the model and encoder load, the application
shows exactly three descending probabilities, a **Why this crop?** explanation,
the expandable global-importance section, exact prediction-input details, and all four
provisional soil statuses. Open **Model & Research Evaluation** and verify stored
metrics and selected charts appear without an experiment starting. The displayed crop
and all numeric results must come from model inference or stored artifacts; no expected
crop or probability is hard-coded in the application.

Also check negative `N`/`P`/`K`, out-of-range pH, and programmatic non-numeric inputs.
They must produce readable validation errors without modifying the dataset or saved
model.
