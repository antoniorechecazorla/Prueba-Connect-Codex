from __future__ import annotations

import argparse
import math
import random
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Math helpers
# -----------------------------


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# -----------------------------
# Labeling / cause adjudication
# -----------------------------


def weather_severity_score(wind_kt: float, vis_km: float, ceiling_ft: float, ts: int) -> float:
    wind_risk = min(wind_kt / 50.0, 1.0)
    vis_risk = 1.0 - min(vis_km / 10.0, 1.0)
    ceil_risk = 1.0 - min(ceiling_ft / 3000.0, 1.0)
    ts_risk = float(ts)
    return max(0.0, min(1.0, 0.30 * wind_risk + 0.30 * vis_risk + 0.25 * ceil_risk + 0.15 * ts_risk))


def atfm_extraordinary_score(atfm_minutes: float, atfm_weather_or_capacity: int) -> float:
    delay_risk = min(math.log1p(max(atfm_minutes, 0.0)) / math.log1p(240.0), 1.0)
    code_risk = float(atfm_weather_or_capacity)
    return max(0.0, min(1.0, 0.7 * delay_risk + 0.3 * code_risk))


def internal_operational_score(prev_leg_delay: float, rotation_slack: float, tech_or_crew_code: int) -> float:
    prev_risk = min(math.log1p(max(prev_leg_delay, 0.0)) / math.log1p(240.0), 1.0)
    slack_risk = 1.0 - min(max(rotation_slack, 0.0) / 90.0, 1.0)
    code_risk = float(tech_or_crew_code)
    return max(0.0, min(1.0, 0.45 * prev_risk + 0.35 * slack_risk + 0.20 * code_risk))


def build_label(row: Dict[str, float]) -> Tuple[Optional[int], float, str, float]:
    outcome = (row["arr_delay_min"] >= 180.0) or (row["cancelled"] == 1)
    if not outcome:
        return 0, 1.0, "not_qualifying_outcome", 0.0

    wx = weather_severity_score(row["wind_kt"], row["vis_km"], row["ceiling_ft"], int(row["ts_flag"]))
    atfm = atfm_extraordinary_score(row["atfm_min"], int(row["atfm_weather_or_capacity"]))
    notam = float(row["notam_major_disruption"])
    internal = internal_operational_score(
        row["prev_leg_delay_min"],
        row["rotation_slack_min"],
        int(row["tech_or_crew_code"]),
    )

    s_ext = 0.45 * wx + 0.40 * atfm + 0.15 * notam
    s_int = 0.60 * internal + 0.40 * float(row["airline_code_nonextra"])
    p_extra = sigmoid(3.5 * (s_ext - s_int))

    if p_extra >= 0.70:
        return 0, p_extra, "extraordinary_excluded", p_extra
    if p_extra <= 0.30:
        return 1, 1.0 - p_extra, "qualifying_non_extra", p_extra
    return None, 0.5, "uncertain_cause", p_extra


# -----------------------------
# Compensation / EV
# -----------------------------


def compensation_tier(distance_km: float, intra_eu: int) -> int:
    if distance_km <= 1500:
        return 250
    if intra_eu == 1 and distance_km > 1500:
        return 400
    if 1500 < distance_km <= 3500:
        return 400
    return 600


def expected_compensation(distance_km: float, intra_eu: int, p_half_reduction: float) -> float:
    base = compensation_tier(distance_km, intra_eu)
    reduction_factor = 1.0 - 0.5 * p_half_reduction
    return base * reduction_factor


# -----------------------------
# Synthetic data generator
# -----------------------------


def rand_choice(rng: random.Random, items: Sequence[str]) -> str:
    return items[rng.randrange(0, len(items))]


def generate_synthetic_flights(n: int, seed: int = 42) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    airlines = ["FR", "U2", "LH", "AF", "IB", "KL", "W6", "VY"]
    origins = ["CDG", "FRA", "AMS", "MAD", "BCN", "MXP", "DUB", "WAW"]
    dests = ["FCO", "BER", "LIS", "VIE", "ATH", "CPH", "PRG", "BRU"]
    aircraft = ["A320", "A321", "B738", "E190", "A319"]

    rows: List[Dict[str, float]] = []
    for _ in range(n):
        row: Dict[str, float] = {
            "airline": rand_choice(rng, airlines),
            "origin": rand_choice(rng, origins),
            "dest": rand_choice(rng, dests),
            "aircraft": rand_choice(rng, aircraft),
            "dep_hour": rng.randint(0, 23),
            "dow": rng.randint(0, 6),
            "month": rng.randint(1, 12),
            "distance_km": rng.uniform(300, 4200),
            "intra_eu": rng.randint(0, 1),
            "ticket_price": rng.uniform(35, 380),
            "prev_leg_delay_min": max(0.0, min(260.0, rng.gauss(35, 40))),
            "rotation_slack_min": max(5.0, min(120.0, rng.gauss(50, 20))),
            "wind_kt": max(0.0, min(65.0, rng.gauss(16, 10))),
            "vis_km": max(0.1, min(15.0, rng.gauss(8, 3))),
            "ceiling_ft": max(100.0, min(8000.0, rng.gauss(3200, 1300))),
            "ts_flag": 1 if rng.random() < 0.11 else 0,
            "atfm_min": min(260.0, rng.gammavariate(2.1, 13.0)),
            "atfm_weather_or_capacity": 1 if rng.random() < 0.32 else 0,
            "notam_major_disruption": 1 if rng.random() < 0.06 else 0,
            "tech_or_crew_code": 1 if rng.random() < 0.24 else 0,
            "airline_code_nonextra": 1 if rng.random() < 0.50 else 0,
        }

        wx = weather_severity_score(row["wind_kt"], row["vis_km"], row["ceiling_ft"], int(row["ts_flag"]))
        int_risk = internal_operational_score(
            row["prev_leg_delay_min"], row["rotation_slack_min"], int(row["tech_or_crew_code"])
        )
        atm_risk = atfm_extraordinary_score(row["atfm_min"], int(row["atfm_weather_or_capacity"]))

        z_delay = -3.1 + 2.3 * int_risk + 1.8 * wx + 2.0 * atm_risk + (0.6 if row["dep_hour"] >= 18 else 0.0)
        p_delay = sigmoid(z_delay)

        delay_event = 1 if rng.random() < p_delay else 0
        p_cancel = max(0.0, min(0.75, 0.06 + 0.20 * atm_risk + 0.10 * wx))
        cancelled = 1 if rng.random() < p_cancel else 0

        if delay_event == 1:
            arr_delay = max(30.0, min(600.0, rng.gauss(220, 90)))
        else:
            arr_delay = max(0.0, min(180.0, rng.gauss(35, 25)))

        row["cancelled"] = cancelled
        row["arr_delay_min"] = arr_delay

        y, conf, reason, p_extra = build_label(row)
        row["y"] = y
        row["label_conf"] = conf
        row["label_reason"] = reason
        row["p_extra"] = p_extra
        row["D"] = 1 if (row["arr_delay_min"] >= 180.0 or row["cancelled"] == 1) else 0
        row["N"] = 1 if (row["D"] == 1 and row["y"] == 1) else 0
        row["p_half_reduction"] = 0.45 if row["arr_delay_min"] < 240 else 0.15
        rows.append(row)

    return rows


# -----------------------------
# Minimal logistic regression
# -----------------------------


class FeatureEncoder:
    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.cat_map: Dict[str, Dict[str, int]] = {c: {} for c in cat_cols}
        self.num_mean: Dict[str, float] = {}
        self.num_std: Dict[str, float] = {}
        self.dim: int = 0

    def fit(self, rows: List[Dict[str, float]]) -> None:
        for c in self.cat_cols:
            vals = sorted({str(r[c]) for r in rows})
            self.cat_map[c] = {v: i for i, v in enumerate(vals)}

        for c in self.num_cols:
            vals = [float(r[c]) for r in rows]
            m = mean(vals)
            s = pstdev(vals) if len(vals) > 1 else 1.0
            self.num_mean[c] = m
            self.num_std[c] = s if s > 1e-9 else 1.0

        self.dim = len(self.num_cols) + sum(len(self.cat_map[c]) for c in self.cat_cols) + 1  # bias

    def transform_row(self, row: Dict[str, float]) -> List[float]:
        x: List[float] = [1.0]
        for c in self.num_cols:
            x.append((float(row[c]) - self.num_mean[c]) / self.num_std[c])
        for c in self.cat_cols:
            onehot = [0.0] * len(self.cat_map[c])
            key = str(row[c])
            if key in self.cat_map[c]:
                onehot[self.cat_map[c][key]] = 1.0
            x.extend(onehot)
        return x

    def transform(self, rows: List[Dict[str, float]]) -> List[List[float]]:
        return [self.transform_row(r) for r in rows]


class LogisticModel:
    def __init__(self, l2: float = 1e-4, lr: float = 0.03, epochs: int = 40):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.w: List[float] = []

    def fit(self, X: List[List[float]], y: List[int], sample_weight: Optional[List[float]] = None) -> None:
        n = len(X)
        d = len(X[0]) if n else 0
        self.w = [0.0] * d
        sw = sample_weight if sample_weight is not None else [1.0] * n

        for _ in range(self.epochs):
            idx = list(range(n))
            random.shuffle(idx)
            for i in idx:
                xi = X[i]
                yi = y[i]
                wi = sw[i]
                p = sigmoid(dot(self.w, xi))
                err = (p - yi) * wi
                for j in range(d):
                    grad = err * xi[j] + self.l2 * self.w[j]
                    self.w[j] -= self.lr * grad

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [sigmoid(dot(self.w, xi)) for xi in X]


# -----------------------------
# Metrics
# -----------------------------


def roc_auc(y_true: List[int], y_score: List[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        pos_in_tie = sum(pairs[k][1] for k in range(i, j + 1))
        rank_sum += avg_rank * pos_in_tie
        rank += (j - i + 1)
        i = j + 1

    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: List[int], y_score: List[float]) -> float:
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    total_pos = sum(y_true)
    if total_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    ap = 0.0
    for idx in order:
        if y_true[idx] == 1:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / total_pos


def brier(y_true: List[int], y_prob: List[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


# -----------------------------
# Training / scoring
# -----------------------------


def train_two_stage_models(rows: List[Dict[str, float]]) -> Dict[str, object]:
    cut = int(0.8 * len(rows))
    train = rows[:cut]
    valid = rows[cut:]

    cat_cols = ["airline", "origin", "dest", "aircraft"]
    num_cols = [
        "dep_hour",
        "dow",
        "month",
        "distance_km",
        "intra_eu",
        "ticket_price",
        "prev_leg_delay_min",
        "rotation_slack_min",
        "wind_kt",
        "vis_km",
        "ceiling_ft",
        "ts_flag",
        "atfm_min",
        "atfm_weather_or_capacity",
        "notam_major_disruption",
        "tech_or_crew_code",
        "airline_code_nonextra",
    ]

    enc = FeatureEncoder(num_cols=num_cols, cat_cols=cat_cols)
    enc.fit(train)

    X_train = enc.transform(train)
    y_train_d = [int(r["D"]) for r in train]

    m_d = LogisticModel(l2=1e-4, lr=0.02, epochs=55)
    m_d.fit(X_train, y_train_d)

    # Stage B on D==1 and confident labels (exclude uncertain)
    train_n = [r for r in train if int(r["D"]) == 1 and r["y"] is not None]
    X_train_n = enc.transform(train_n)
    y_train_n = [int(r["N"]) for r in train_n]
    w_train_n = [float(r["label_conf"]) for r in train_n]

    m_n = LogisticModel(l2=1e-4, lr=0.02, epochs=55)
    m_n.fit(X_train_n, y_train_n, sample_weight=w_train_n)

    X_valid = enc.transform(valid)
    y_valid_d = [int(r["D"]) for r in valid]
    p_d = m_d.predict_proba(X_valid)

    valid_n = [r for r in valid if int(r["D"]) == 1 and r["y"] is not None]
    X_valid_n = enc.transform(valid_n)
    y_valid_n = [int(r["N"]) for r in valid_n]
    p_n_eval = m_n.predict_proba(X_valid_n) if valid_n else [0.5]

    metrics = {
        "stageA_auc": roc_auc(y_valid_d, p_d),
        "stageA_ap": average_precision(y_valid_d, p_d),
        "stageA_brier": brier(y_valid_d, p_d),
        "stageB_auc": roc_auc(y_valid_n, p_n_eval) if len(valid_n) > 20 else None,
    }

    return {"encoder": enc, "m_d": m_d, "m_n": m_n, "valid": valid, "metrics": metrics}


def score_ev(rows: List[Dict[str, float]], bundle: Dict[str, object]) -> List[Dict[str, float]]:
    enc: FeatureEncoder = bundle["encoder"]  # type: ignore[assignment]
    m_d: LogisticModel = bundle["m_d"]  # type: ignore[assignment]
    m_n: LogisticModel = bundle["m_n"]  # type: ignore[assignment]

    X = enc.transform(rows)
    p_d = m_d.predict_proba(X)
    p_n = m_n.predict_proba(X)

    out = []
    for r, pd_, pn_ in zip(rows, p_d, p_n):
        rr = dict(r)
        rr["p_d"] = pd_
        rr["p_n"] = pn_
        rr["p_q"] = pd_ * pn_
        rr["exp_comp"] = expected_compensation(rr["distance_km"], int(rr["intra_eu"]), rr["p_half_reduction"])
        rr["ev"] = rr["p_q"] * rr["exp_comp"] - rr["ticket_price"]
        out.append(rr)
    return out


# -----------------------------
# Optimization and Monte Carlo
# -----------------------------


def select_portfolio_knapsack(rows: List[Dict[str, float]], budget: float, max_n: int) -> List[Dict[str, float]]:
    cands = sorted([r for r in rows if r["ev"] > 0], key=lambda r: r["ev"], reverse=True)[:max_n]
    if not cands:
        return []

    B = int(round(budget))
    n = len(cands)
    prices = [int(round(r["ticket_price"])) for r in cands]
    values = [r["ev"] for r in cands]

    dp = [[0.0] * (B + 1) for _ in range(n + 1)]
    keep = [[0] * (B + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = prices[i - 1]
        v = values[i - 1]
        for b in range(B + 1):
            if w <= b and dp[i - 1][b - w] + v > dp[i - 1][b]:
                dp[i][b] = dp[i - 1][b - w] + v
                keep[i][b] = 1
            else:
                dp[i][b] = dp[i - 1][b]

    b = B
    picked: List[Dict[str, float]] = []
    for i in range(n, 0, -1):
        if keep[i][b] == 1:
            picked.append(cands[i - 1])
            b -= prices[i - 1]

    return sorted(picked, key=lambda r: r["ev"], reverse=True)


def monte_carlo_portfolio(portfolio: List[Dict[str, float]], trials: int = 5000, seed: int = 42) -> Dict[str, float]:
    if not portfolio:
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "prob_positive": 0.0,
            "var_05": 0.0,
            "cvar_05": 0.0,
        }

    rng = random.Random(seed)
    returns: List[float] = []
    for _ in range(trials):
        total = 0.0
        for r in portfolio:
            if rng.random() < r["p_q"]:
                total += r["exp_comp"] - r["ticket_price"]
            else:
                total += -r["ticket_price"]
        returns.append(total)

    sorted_ret = sorted(returns)
    idx_05 = max(0, int(0.05 * len(sorted_ret)) - 1)
    var_05 = sorted_ret[idx_05]
    tail = [x for x in returns if x <= var_05]
    cvar_05 = mean(tail) if tail else var_05

    return {
        "mean_return": mean(returns),
        "std_return": pstdev(returns),
        "prob_positive": sum(1 for x in returns if x > 0) / len(returns),
        "var_05": var_05,
        "cvar_05": cvar_05,
    }


def run_demo(samples: int, budget: float, max_n: int, seed: int) -> None:
    rows = generate_synthetic_flights(samples, seed=seed)
    bundle = train_two_stage_models(rows)
    scored = score_ev(bundle["valid"], bundle)

    portfolio = select_portfolio_knapsack(scored, budget=budget, max_n=max_n)
    risk = monte_carlo_portfolio(portfolio, trials=6000, seed=seed)

    print("=== Metrics ===")
    for k, v in bundle["metrics"].items():
        print(f"{k}: {v}")

    print("\n=== Top selected flights ===")
    if not portfolio:
        print("No positive-EV flights selected under constraints.")
    else:
        for i, r in enumerate(portfolio[:12], start=1):
            print(
                f"{i:02d}. {r['airline']} {r['origin']}->{r['dest']} {r['aircraft']} | "
                f"price={r['ticket_price']:.1f} p_q={r['p_q']:.3f} comp={r['exp_comp']:.1f} ev={r['ev']:.1f}"
            )

    print("\n=== Monte Carlo risk ===")
    for k, v in risk.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functional EC261 prototype (no external dependencies)")
    parser.add_argument("--samples", type=int, default=3500)
    parser.add_argument("--budget", type=float, default=1200.0)
    parser.add_argument("--max-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_demo(samples=args.samples, budget=args.budget, max_n=args.max_n, seed=args.seed)
