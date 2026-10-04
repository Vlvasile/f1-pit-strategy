# F1 Pit-Stop Strategy: When Should a Driver Change Tyres?

An end-to-end data science project on Formula 1 race data (2022–2025). I used lap timing data to measure how tyres wear out, how much faster the car gets as fuel burns, and then tried to answer one question:

> **On which lap should a driver pit to lose the least time over the whole race?**

The project goes from raw data download to cleaning, analysis, machine learning models, experiment tracking with MLflow and a simple strategy simulator. At the end I compare the strategy predictions with what real F1 teams did in 2025.

---

## Results in short

- Collected and cleaned **92 races (2022–2025)**, about **78,000 race laps** used for analysis.
- Separated two effects that are always mixed in lap times: **fuel burn makes the car ~0.05 s/lap faster**, while **tyre wear makes it 0.03–0.055 s/lap slower** depending on the compound.
- Best lap-time model (regularised LightGBM) predicts 2025 lap times with **0.55 s mean error**, about **30% better** than a naive baseline.
- A simple tyre-wear formula predicts the real 2025 pit lap with a **median error of ~3.5 laps** (8 of 12 circuits within 5 laps).
- Interesting finding: the model with the best lap-time accuracy (LightGBM) was **not** the best for strategy. A simple physics-style formula worked better.

---

## Data

- Source: [FastF1](https://docs.fastf1.dev) Python library (official F1 live timing data).
- Seasons: 2022, 2023, 2024, 2025, race sessions only.
- One row = one lap of one driver.

FastF1 has an API limit of 500 requests per hour, so downloading all seasons took several runs. I made the download script save every race to its own file and skip races that already exist, so it can continue after hitting the limit.

---

## Data cleaning

Not every lap is useful for measuring tyre wear. I removed:

| Removed | Why |
|---|---|
| First lap | standing start, not a normal lap |
| Pit in / pit out laps | include time in the pit lane |
| Laps under Safety Car, VSC or yellow flags | everyone drives slowly |
| Inaccurate, deleted or FastF1-generated laps | unreliable timing |
| Intermediate / wet tyres | the project focuses on dry strategy |
| Laps more than 5% slower than the median of their stint | traffic, mistakes, damage |

```
83,684 clean laps (after per-race cleaning)
   ↓ dry tyres only (SOFT / MEDIUM / HARD), known tyre age
78,633
   ↓ outlier filter (>5% slower than stint median)
78,233 laps used for analysis and modelling
```

A "stint" in this project means the part of a race a driver does on one set of tyres (between pit stops).

---

## Exploratory analysis

### Q1. How fast does each tyre compound degrade?

![Degradation curve](images/degradation_curve.png)

For every stint I fitted a line "tyre age → lap time" and took the median slope (3,646 stints with at least 8 laps):

| Compound | Net change per lap of tyre age |
|---|---|
| SOFT | +0.030 s |
| MEDIUM | +0.006 s |
| HARD | −0.005 s |

SOFT is the only compound where laps clearly get slower during a stint. On HARD the laps even get faster. This does not mean HARD tyres don't wear — it means the fuel effect is bigger than the wear (see Q2).

![Degradation by stint length](images/degradation_by_stint_length.png)

Splitting stints by length showed something important: short stints (8–24 laps) lose 0.2–0.5 s before the pit stop, so tyre wear is what forces teams to stop. Long stints (25+ laps) get faster all the time — they come from races where tyres barely wear. So the right side of the first chart is biased by these "lucky" long stints (survivorship bias).

### Q2. How much does fuel burn help, and what is the real tyre wear?

Inside one stint, lap number and tyre age grow together, so their effects are mixed. After a pit stop tyre age resets but lap number keeps going. I used a linear regression with race-driver fixed effects to separate them:

| Effect | Seconds per lap |
|---|---|
| Fuel burn + track evolution | **−0.051** |
| SOFT wear | +0.055 |
| MEDIUM wear | +0.037 |
| HARD wear | +0.031 |

![Fuel vs wear](images/fuel_vs_wear.png)

Over 30 laps on MEDIUM, tyre wear costs about 1.1 s per lap, but the lighter car gains about 1.5 s, so the driver ends up faster.

![SOFT vs HARD crossover](images/soft_vs_hard_crossover.png)

A fresh SOFT is about 0.36 s faster than a fresh HARD, but after about **15 laps** a worn SOFT becomes slower than a HARD of the same age.

### Q3. Which circuits are hardest on tyres?

I ran the same regression separately for every circuit (fuel effect estimated per circuit too):

![Wear by circuit](images/wear_by_circuit.png)

- Highest wear: **Bahrain** (0.129 s/lap), Spain, Belgium, Japan
- Lowest wear: Australia (0.004 s/lap), Saudi Arabia, Qatar, Monaco

The fuel + track evolution effect is also very different between circuits: from −0.032 s/lap in Austria (short lap) to −0.091 s/lap in Belgium (long lap). Using one global value made street circuits like Las Vegas look like they had almost no tyre wear.

### Q4. Do high-wear circuits need more pit stops?

![Wear vs stops](images/wear_vs_stops.png)

Yes — circuits with higher tyre wear have more pit stops per driver (from ~1.1 at Saudi Arabia, Miami, Baku and Monza to ~2.4 at Bahrain and Spain). The biggest outliers are explained by other things: stint length limits (Qatar), rain and red flags (São Paulo, Netherlands, Australia).

---

## Lap-time model

**Target:** how much slower or faster a lap is compared to the driver's typical lap in that race (`LapDelta`). This removes the difference between circuits and cars, so the model focuses on tyres and fuel.

**Features:** tyre age, compound, lap number, circuit, year, team, new/used tyre set.

**Split by season:** train 2022–2024, test 2025 (for tuning: train 2022–2023, validation 2024). A random split would leak laps from the same race into both train and test.

| Model | Train MAE (s) | Test MAE (s) | vs naive |
|---|---|---|---|
| Naive (always predict 0) | – | 0.796 | – |
| Linear regression | 0.587 | 0.594 | −25% |
| LightGBM (default settings) | 0.396 | 0.603 | −24% |
| **LightGBM (regularised)** | 0.526 | **0.554** | **−30%** |
| LightGBM (regularised + monotone) | 0.535 | 0.553 | −31% |

The default LightGBM memorised the training seasons (train 0.40 vs test 0.60). With stronger regularisation (simpler trees, at least 200 laps per leaf, subsampling) and early stopping on the 2024 season, the model stopped after 77 trees and the train–test gap dropped from 0.21 s to 0.03 s.

The monotone version adds two rules from physics: an older tyre can't make the lap faster, and a later lap (less fuel) can't make it slower.

All experiments are tracked in **MLflow** (SQLite backend), and the final models are registered in the MLflow Model Registry as `lap-time-lgbm`.

---

## Pit-stop strategy

### Pit-stop time loss

From raw FastF1 data I calculated how much time a pit stop costs on each circuit: in-lap + out-lap minus two normal laps, only green-flag stops (2024–2025). It ranges from **19 s in Monaco** to **30 s at Silverstone**.

### When to pit

In a dry race every driver must use two different compounds, so one pit stop is mandatory. That means the pit-stop cost is the same no matter which lap you stop on. The real question is **how to split the race between two sets of tyres** so the total tyre wear is as small as possible.

I tried two approaches for a one-stop MEDIUM → HARD race:

1. **Simulator with the LightGBM model** — predict every lap of the race for every possible pit lap and pick the fastest total.
2. **Simple formula** using the wear rates from Q2:

```
pit lap ≈ (HARD wear × race laps + pace gap) / (MEDIUM wear + HARD wear)
```

### Comparison with real 2025 races

![Predicted vs real pit lap](images/predicted_vs_real_pit.png)

| Method | Mean error vs real pit lap |
|---|---|
| LightGBM v1 simulator | 17.3 laps |
| LightGBM v2 (monotone) simulator | 12.6 laps |
| **Simple formula** | **6.0 laps** |

When I compared only with green-flag pit stops (no Safety Car stops, which are a different situation), the formula had a **median error of ~3.5 laps**, and 8 of 12 circuits were within 5 laps (for example Austria 32.4 vs 32.5, Hungary 32.4 vs 33).

**Why the LightGBM simulator was worse:** tree models can't extrapolate. Long stints (30+ laps) are rare in the data, so the model predicts almost no extra wear for very old tyres and tells the driver to stay out too long. The model with the best lap-by-lap accuracy was not the best for the actual decision.

**Why teams still pit earlier than the formula (about 3–4 laps on average):** the formula only knows about tyres. Real teams also think about the *undercut* (pitting a lap earlier than a rival to pass them on fresh tyres), traffic and Safety Car chances. The bottom of the "time lost vs pit lap" curve is very flat — pitting 5 laps early costs less than 1 second over the race — so these strategic reasons easily win.

I also tried tyre wear estimated separately for every circuit. It made the predictions **worse** (mean error 6.6 vs 5.0 laps), because with only 2–3 races per circuit the estimates were very noisy (some even negative). The global formula generalises better.

---

## Limitations

- Only dry races and only one-stop MEDIUM → HARD strategies are evaluated.
- The model knows nothing about other cars: no undercut, traffic or Safety Car.
- Fuel effect and track evolution (rubber building up on the track) can't be separated with this data.
- Some teams changed names between seasons (e.g. AlphaTauri → RB → Racing Bulls), which the model sees as different teams.
- Real pit laps in the evaluation come from a small number of drivers per circuit.

---

## Project structure

```
f1-pit-strategy/
├── notebooks/
│   ├── 01_load_data.ipynb   # first look at one race (Monza 2024), cleaning logic
│   ├── 02_ingest.ipynb      # download all races 2022–2025
│   ├── 03_clean.ipynb       # data checks, dry tyres, outliers → laps_model.parquet
│   ├── 04_eda.ipynb         # questions Q1–Q4
│   └── 05_model.ipynb       # models, MLflow, pit loss, strategy simulator
├── src/
│   ├── clean.py             # lap cleaning function
│   └── ingest.py            # download and save races (resumes after API limit)
├── images/                  # charts used in this README
├── requirements.txt
└── .gitignore               # data/, cache/, mlruns/ and mlflow.db are not in the repo
```

---

## How to run

```bash
# 1. Create an environment (Python 3.11)
conda create -n f1 python=3.11 -y
conda activate f1
pip install -r requirements.txt

# 2. Run the notebooks in order
#    02_ingest.ipynb  -> downloads the data (needs several runs because of the API limit)
#    03_clean.ipynb   -> builds data/processed/laps_model.parquet
#    04_eda.ipynb     -> analysis
#    05_model.ipynb   -> models and strategy

# 3. Open the MLflow UI (from the project root)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The data is not stored in the repository — it is downloaded by the notebooks.

---

## Tech stack

Python · FastF1 · pandas · NumPy · scikit-learn · LightGBM · Plotly · MLflow · Parquet · Git

---

## Next steps

- Move the strategy formula into `src/strategy.py`
- FastAPI service: circuit + race length → optimal pit lap and pit window
- Docker image for the API
- Tests and GitHub Actions
- Add Safety Car probability and two-stop strategies
