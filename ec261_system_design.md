# EC261 Predictive Eligibility & Expected-Value System (EU Flights)

## 1) Problem formulation

### 1.1 Binary target
Define per operated flight instance \(i\):

\[
y_i = \mathbb{1}\Big((\text{arr\_delay}_i \ge 180\,\text{min} \;\lor\; \text{cancelled}_i = 1) \land (\text{extraordinary}_i = 0)\Big)
\]

Where:
- `arr_delay` = actual arrival timestamp − scheduled arrival timestamp (in gate-in terms when possible).
- `cancelled` = flight canceled before operation (exclude pure schedule changes unless legally treated as cancellation in your legal rules engine).
- `extraordinary` = event outside airline’s normal activity and beyond actual control, with reasonable measures unable to avoid outcome.

### 1.2 Operational definition of extraordinary circumstances
Use a hierarchical ruleset plus confidence score:

**Likely extraordinary** (default extraordinary=1):
1. Severe weather at origin/destination/en-route windows:
   - Thunderstorm/convective cells, freezing rain, snowstorm, low-visibility below minima, crosswind beyond aircraft/airport limits.
2. ATC flow restrictions outside airline control:
   - Eurocontrol ATFM regulation with weather/ATC staffing/sector capacity causes.
3. Airspace closure / security risks:
   - NOTAM-driven closure, military activity, security incidents.
4. Airport closure/incidents:
   - Runway closure due to contamination, bird-strike runway inspection closures.
5. External industrial action:
   - ATC/airport handling/general strike not attributable to airline staff.

**Likely non-extraordinary** (default extraordinary=0):
1. Technical defects from wear-and-tear/maintenance planning.
2. Crew scheduling mismanagement or late crew.
3. Airline operational disruptions (rotation misconnect, turn-around delays, fueling/catering internal).
4. Commercial/operational cancellations for fleet balancing.

**Ambiguous / needs reconciliation**:
- Mixed-cause delays (e.g., initial ATFM then knock-on rotation).
- Airline-reported code conflicts with exogenous evidence.

### 1.3 Label-noise mitigation by cross-validation with external evidence
Airline cause codes are noisy and strategic; create a **cause adjudication model**:

Inputs for each delayed/cancelled flight:
- Airline IATA delay/cancel code + text reason.
- METAR/TAF-derived weather severity indices at relevant times.
- Eurocontrol ATFM delay minutes and reason category per flight/airport/sector.
- NOTAM events near O/D airports and route FIRs.

Build an evidence score:
\[
S_{ext} = w_1\cdot \text{weather\_sev} + w_2\cdot \text{atfm\_extra\_signal} + w_3\cdot \text{notam\_closure\_signal}
\]
and internal-control score:
\[
S_{int} = v_1\cdot \text{prev\_leg\_late} + v_2\cdot \text{airline\_rotational\_stress} + v_3\cdot \text{tech/crew\_codes}
\]
Set extraordinary with confidence:
\[
\Pr(\text{extraordinary}=1)=\sigma(S_{ext}-S_{int})
\]
Then hard label via thresholds (e.g., >0.7 extraordinary, <0.3 non-extraordinary, else uncertain for weak supervision / down-weighting).

### 1.4 Pseudocode for robust label construction
```python
def build_label(flight):
    # Raw outcome condition
    outcome = (flight.arr_delay_min >= 180) or (flight.cancelled == 1)
    if not outcome:
        return 0, 1.0, "not_qualifying_outcome"

    # External evidence
    wx = weather_severity_score(flight.metar, flight.taf)            # 0..1
    atfm = atfm_extraordinary_score(flight.atfm_codes, flight.atfm_min) # 0..1
    notam = notam_disruption_score(flight.notam_events)             # 0..1

    # Internal/control evidence
    internal_ops = internal_operational_score(
        prev_leg_delay=flight.prev_leg_delay,
        rotation_slack=flight.rotation_slack_min,
        airline_code=flight.airline_cause_code,
        turnaround_buffer=flight.turnaround_buffer_min
    )  # 0..1

    # Cause reconciliation
    s_ext = 0.45*wx + 0.40*atfm + 0.15*notam
    s_int = 0.60*internal_ops + 0.40*internal_code_nonextra_score(flight.airline_cause_code)
    p_extra = sigmoid(3.5*(s_ext - s_int))

    if p_extra >= 0.70:
        extraordinary = 1
        conf = p_extra
    elif p_extra <= 0.30:
        extraordinary = 0
        conf = 1.0 - p_extra
    else:
        # uncertain -> keep soft label or lower training weight
        extraordinary = None
        conf = 0.5

    if extraordinary == 0:
        return 1, conf, "qualifying_non_extra"
    elif extraordinary == 1:
        return 0, conf, "extraordinary_excluded"
    else:
        return None, conf, "uncertain_cause"
```

---

## 2) Expected Value (EV) framework

### 2.1 Core objective
For candidate ticket \(f\):
\[
EV(f)=\mathbb{P}(Q_f=1)\cdot C_f - P_f
\]
Where:
- \(Q_f=1\): EC261-qualifying event (>=3h or cancellation and non-extraordinary).
- \(C_f\): expected compensation after EC261 distance tier and reduction rules.
- \(P_f\): ticket price paid.

### 2.2 EC261 compensation tiers
By great-circle distance and intra-EU logic:
- €250: flights up to 1500 km.
- €400: intra-EU >1500 km and other flights 1500–3500 km.
- €600: flights >3500 km (with specific long-haul conditions).

Reduction rule (typical): 50% reduction if re-routing arrival delay vs scheduled is within threshold bands (commonly 2/3/4 hours by tier class). Model this via expected payout factor \(r_f\in\{1,0.5\}\) or probabilistic \(\mathbb{E}[r_f]\).

Thus:
\[
C_f = C^{tier}_f\cdot \mathbb{E}[r_f \mid Q_f=1]
\]

### 2.3 Probability decomposition
\[
\mathbb{P}(Q_f=1)=\mathbb{P}(D_f=1)\cdot\mathbb{P}(N_f=1\mid D_f=1)
\]
Where:
- \(D_f=1\): delay >=3h or cancellation.
- \(N_f=1\): non-extraordinary cause.

Estimate with two-stage modeling:
1. **Stage A**: \(\hat p_D=\Pr(D=1\mid X)\).
2. **Stage B**: \(\hat p_N=\Pr(N=1\mid D=1, X, Z_{evidence})\).
3. \(\hat p_Q=\hat p_D\cdot\hat p_N\).

This improves interpretability and allows independent recalibration when cause adjudication updates.

---

## 3) Dataset and feature engineering

### 3.1 Feature groups

1. **Route & geometry**
   - Origin/destination ICAO/IATA, country, airport class.
   - Great-circle distance, block-time percentile, route directionality.

2. **Carrier & aircraft**
   - Airline, operating carrier, alliance, LCC/legacy indicator.
   - Aircraft type (A320, B738...), age proxy, seat density proxy.

3. **Schedule/time**
   - STD hour, weekday, week-of-year, month, holiday indicator.
   - Cyclic encoding for hour/day-of-year:
     \(\sin(2\pi h/24), \cos(2\pi h/24)\), etc.

4. **Historical performance (leakage-safe rolling windows)**
   - Route delay>=3h rate over last 7/30/90 days.
   - Route cancellation rate over last 30/90 days.
   - Airline-on-route percentile vs network baseline.

5. **Aircraft rotation (critical)**
   - Previous leg arrival delay.
   - Scheduled turn time minus minimum turn requirement (slack).
   - Number of legs already flown same tail same day.

6. **Weather**
   - METAR features at t-3h..t+1h around STD/STA:
     wind/gust, visibility, ceiling, precipitation type/intensity,
     CB/TS flags, temperature/dew spread.

7. **Congestion / ATM**
   - ATFM regulation minutes at departure airport, destination airport, relevant sectors.
   - Sector load ratio, declared capacity reductions.
   - Departure queue proxy (recent off-block delays).

8. **NOTAM/operations disruptions**
   - Runway closures, nav-aid outages, de-icing operations, strike notices.

### 3.2 Real-world data sources/APIs
- **Eurocontrol NM B2B**: ATFM regs, delay reasons, network ops.
- **METAR/TAF**: NOAA/OGIMET/airport feeds.
- **NOTAM**: EAD/FAA mirrored feeds (for EU airports use authoritative EAD providers).
- **Flight operations data**:
  - OpenSky Network (tracking context),
  - AviationStack/FlightAware/OAG/Cirium (commercial schedule/status feeds),
  - EUROSTAT/airport movement datasets for aggregates.
- **Airport metadata**: OurAirports, AIP-derived reference data.

### 3.3 Missing data strategy
- Weather missing:
  - short gap (<2 intervals): time interpolation;
  - larger: nearest station with distance penalty + missingness flag.
- Aircraft tail missing:
  - fallback to flight-number rotation heuristics + uncertainty feature.
- ATFM missing:
  - impute by airport-hour historical median + flag.
- Cause label uncertainty:
  - soft labels/weights in training.

### 3.4 Transformations
- Log transforms: heavy-tail variables (ATFM minutes, prior delay minutes): \(\log(1+x)\).
- Scaling: robust scaler for linear models.
- Encoding:
  - target/frequency encoding for high-cardinality categorical (airline, route).
  - one-hot for low-cardinality bins.
- Interaction terms:
  - airport × hour, airline × airport, weather × airport resilience.

---

## 4) Predictive modeling

### 4.1 Logistic regression baseline (production-grade baseline)
Pipeline steps:
1. Split by **time** (rolling-origin validation).
2. Preprocess with `ColumnTransformer`:
   - numeric: impute + scale.
   - cyclic features via custom transformer.
   - high-cardinality categoricals via target encoding (fold-safe).
3. Handle imbalance:
   - class weights (`balanced`) and/or focal threshold tuning.
   - SMOTE only on training folds and only if feature space supports it.
4. Calibrate probabilities (`IsotonicRegression`/Platt).

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Assume custom transformers: CyclicTimeEncoder, FoldSafeTargetEncoder
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cyc", CyclicTimeEncoder(), time_cols),
    ("cat_te", FoldSafeTargetEncoder(cols=high_card_cols, smooth=20), high_card_cols),
], remainder="drop")

base = LogisticRegression(
    penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, n_jobs=-1
)

clf = Pipeline([
    ("pre", pre),
    ("model", CalibratedClassifierCV(base, method="isotonic", cv=3))
])

clf.fit(X_train, y_train, sample_weight=w_train)  # w_train from label confidence
p = clf.predict_proba(X_valid)[:, 1]
```

### 4.2 Gradient boosting extension (LightGBM)
- Handles nonlinear interactions and missing values naturally.
- Use categorical support or target encoding.
- Objective `binary`, optimize AUC + logloss, then calibrate.

### 4.3 Optional survival framing
Model time-to-arrival-delay-threshold exceedance with survival/hazard models for dynamic updates pre-departure and en-route.

### 4.4 Optional causal weather adjustment
Use doubly robust estimation / causal forests to separate route structural risk from transient weather shocks to avoid overfitting weather regimes.

### 4.5 Interpretation
- Logistic coefficient \(\beta_j\): one-unit increase in feature changes log-odds by \(\beta_j\).
- Odds ratio \(e^{\beta_j}\): multiplicative effect on odds.
- Typical strongest predictors:
  1. prior leg delay/rotation slack,
  2. ATFM minutes/regulations,
  3. weather severity at hubs,
  4. departure bank hour (late-day propagation),
  5. airline-route historical reliability.

---

## 5) Decision engine (optimization layer)

For candidate set \(\mathcal{F}\):
\[
EV_f = \hat p_{Q,f}\cdot C_f - P_f
\]

### 5.1 Constraints
- Budget: \(\sum_f P_f x_f \le B\)
- Departure airports subset: \(o_f \in \mathcal{O}\)
- Time window: \(t_f \in [t_0,t_1]\)
- Optional diversification: cap airline/airport concentration.

### 5.2 Selection algorithms
1. **Simple filter + sort**:
   - keep flights with \(EV_f>0\), confidence >= threshold, liquidity constraints.
   - rank by risk-adjusted value: \(EV_f / \sigma_f\) or \(EV_f\).
2. **0/1 knapsack** (portfolio choice):
\[
\max_x \sum_f EV_f x_f\quad s.t.\quad \sum_f P_f x_f \le B,\; x_f\in\{0,1\}
\]
3. **Chance-constrained variant** using Monte Carlo-estimated downside risk.

---

## 6) Monte Carlo simulation

For each selected flight \(f\), payout outcome:
\[
R_f = \begin{cases}
C_f - P_f & \text{with prob } \hat p_{Q,f}\\
-P_f & \text{with prob } 1-\hat p_{Q,f}
\end{cases}
\]
Portfolio return per trial:
\[
R^{(k)} = \sum_{f=1}^{N} R_f^{(k)}
\]

Algorithm:
1. Choose portfolio of \(N\) flights.
2. For each trial \(k=1..T\): sample Bernoulli(\(\hat p_{Q,f}\)) for each \(f\), compute \(R^{(k)}\).
3. Report mean, variance, VaR/CVaR, probability \(R>0\).

Law of large numbers: as \(N\) increases with weak dependence, empirical average converges to expected return; variance per-flight averages down, but correlation (weather/systemic ATC shocks) slows diversification.

---

## 7) Extracting optimal flight profiles

Use grouped EV analytics and SHAP/partial dependence:

- Aggregate by `(route, airline, aircraft_family, dep_hour_bin)`.
- Compute:
  - mean predicted \(\hat p_Q\), mean EV,
  - realized calibration-adjusted EV,
  - risk metrics (std/CVaR).

Expected structural patterns:
1. **Medium-haul** often sweet spot (higher compensation than short-haul with manageable ticket cost).
2. **Late-day rotations** show delay propagation (higher \(p_D\)).
3. **Congested hubs** increase disruption probability; secondary airports may reduce systemic ATC delays but can have recovery limitations.
4. **Carrier model effects**:
   - tight-turn LCC networks: stronger knock-on risk,
   - legacy carriers: more buffers but hub wave peaks can spike delays.

---

## 8) Production system architecture

### 8.1 Logical layers
1. **Ingestion**
   - Batch: schedules/fares/history (hourly/daily pulls).
   - Streaming: live flight status, METAR, ATFM updates.
2. **Storage**
   - PostgreSQL (OLTP/reference),
   - Data lake (S3/MinIO parquet),
   - Feature store (Feast): offline parquet + online Redis.
3. **Processing**
   - Airflow for batch DAGs,
   - Kafka + Flink/Spark Structured Streaming for low-latency feature updates.
4. **Modeling/serving**
   - Python training pipelines, MLflow registry.
   - FastAPI scoring service (real-time inference).
5. **Decision engine**
   - Optimization service (knapsack/ranking), cached results in Redis.
6. **UI/API**
   - Dashboard (React + Plotly/Apache ECharts).

### 8.2 Feature store design
- **Offline store**: training point-in-time correct joins.
- **Online store**: keyed by `(flight_instance_id, timestamp)` for current scoring.
- Feature freshness SLAs (e.g., weather <=10 min lag).

### 8.3 MLOps
- MLflow: experiment tracking, model versioning, stage transitions.
- Monitoring:
  - data drift (PSI/KS),
  - concept drift (calibration decay),
  - business KPI drift (realized EV gap).
- Retraining cadence:
  - weekly full retrain + daily incremental calibration.

### 8.4 Real-time scoring loop
1. Event arrives (status/weather/ATFM).
2. Stream processor updates online features.
3. Scoring API recomputes \(\hat p_D, \hat p_N, \hat p_Q, EV\).
4. Decision engine reranks candidate flights.
5. Dashboard updates and triggers alerts.

---

## 9) Output / product design

Dashboard tables/cards:
- Ranked flights by:
  - `P(EC261 qualifying)`
  - expected compensation
  - ticket price
  - EV and risk-adjusted EV
- Confidence band (model + label uncertainty).
- Explainability panel:
  - SHAP top contributors (e.g., prior leg +70 min, ATFM reg +45 min).
- Scenario controls:
  - budget slider, airport filters, date/time windows.

---

## 10) Evaluation strategy

### 10.1 Statistical metrics
- ROC-AUC for discrimination.
- PR-AUC for rare qualifying outcomes.
- Brier score + reliability curves for calibration.

### 10.2 Business metric (primary)
- **Out-of-sample realized EV**:
\[
\widehat{EV}_{realized}=\frac{1}{M}\sum_{m=1}^{M}(\text{realized\_payout}_m-\text{ticket\_price}_m)
\]
Evaluate by policy bucket and confidence decile.

### 10.3 Threshold optimization by EV
Choose decision threshold \(\tau\) maximizing validation EV:
\[
\tau^*=\arg\max_\tau \sum_{f:\hat p_{Q,f}\ge\tau}(\hat p_{Q,f}C_f-P_f)
\]
Tradeoff:
- Higher precision: fewer false positives, lower volume.
- Higher recall: more opportunities but includes lower-EV flights.

---

## 11) Limitations

1. **Data gaps**
   - Tail-level rotations may be missing/noisy.
   - True legal claim outcomes not always available.
2. **Cause ambiguity**
   - Extraordinary vs non-extraordinary can remain disputable.
3. **Rare events / imbalance**
   - Qualifying events are sparse; variance high for small samples.
4. **Regime shifts**
   - Network disruptions, labor cycles, policy changes alter priors.
5. **Correlation risk**
   - Weather/ATC shocks create clustered losses; naive IID assumptions optimistic.

---

## Implementation blueprint (condensed)
1. Build robust labeler with external evidence reconciliation + confidence weights.
2. Train two-stage calibrated models (`D` and `N|D`) with time-split validation.
3. Compute per-flight compensation expectation and EV.
4. Select portfolio via constrained optimization.
5. Validate with Monte Carlo and backtests on rolling windows.
6. Deploy streaming feature + real-time scoring + monitoring loop.
