# PerCoF

Code for my paper "PerCoF: Personalised Counterfactual Explanations with Metaheuristics" (University of Antwerp).

PerCoF looks for counterfactuals that respect what a rejected applicant says they can change (which features, direction, max change, ranking). If nothing fits, it relaxes their limits step by step. I compare GA, NSGA-II, simulated annealing and random search on the German Credit dataset, with simulated users.

## Running it

```
pip install -r requirements.txt
cd percof
```

1. `prep_credit.ipynb` - data prep, random forest, saves `RF.pkl` and `percof_setup.pkl`
2. `python users.py` - generates the simulated user profiles (`profiles.pkl`)
3. `python experiments.py --workers 14 --budget 1000` (also ran with 100 and 250)
4. `python analysis.py` - tables and tests in `tables/`
5. `python visualise.py` - figures in `figures/`

Step 3 takes about an hour with 14 workers, mostly because of SA. Results are already included (`results_budget_*.parquet`, `convergence_budget_*.parquet`), so you can skip to step 4.

## Files

- `problem.py` - problem definition (encoding, constraints, objectives)
- `metaheuristics.py` - random search, GA, NSGA-II, SA
- `relaxation.py` - relaxation procedure (bounds first, then reluctant features)
- `users.py` - simulated users, with and without SHAP nudge
- `experiments.py` - runs everything

Run on Python 3.14 with scikit-learn 1.8.0 (the pickled model needs this version).
