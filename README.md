# Prueba-Connect-Codex

## EC261 functional prototype

This repository includes a runnable prototype implementing the core EC261 framework end-to-end:

- robust label construction (delay/cancellation + extraordinary-cause adjudication)
- two-stage predictive modeling (`P(D)` and `P(N|D)`) via in-code logistic models
- EC261 compensation tiers and expected-value scoring
- constrained portfolio selection (knapsack with budget)
- Monte Carlo simulation for return/risk analysis

## Run

```bash
python prototype/ec261_prototype.py --samples 3500 --budget 1200 --max-n 30 --seed 42
```

## Output

The script prints:

- stage A/B model quality metrics (AUC/AP/Brier where applicable)
- selected flights ranked by expected value
- Monte Carlo risk stats (mean/std/probability positive return/VaR/CVaR)

## Notes

- This prototype is self-contained and requires only standard Python 3.
- It uses synthetic flight data generation so it can run offline.
- To productionize, replace `generate_synthetic_flights(...)` with real ETL feeds (Eurocontrol, METAR, NOTAM, flight status, fares).
