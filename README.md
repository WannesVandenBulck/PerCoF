# PerCoF

Code for my paper "PerCoF: Personalised Counterfactual Explanations with Metaheuristics" for the metaheusristics PhD-course.

PerCoF is a defined as a personalised countefactual explanation problem and looks for counterfactuals that respect what a rejected applicant says they can/want to change (which features, direction, max change, ranking). If nothing fits, it relaxes their limits step by step, first by relaxing the max change and later by adding features the user provided via a relucantcy ranking. I compare GA, NSGA-II, simulated annealing and random search on the German Credit dataset, with simulated user preferences.

## Running the repo

```
pip install -r requirements.txt
cd percof
```

1. `prep_credit.ipynb` - data prep, random forest, saves `RF.pkl` and `percof_setup.pkl`
2. `python users.py` - generates the simulated user preferences (`profiles.pkl`)
3. `python experiments.py --workers 14 --budget 1000` (also ran with 100 and 250 budgets)
4. `python analysis.py` - tables and tests in `tables/`
5. `python visualise.py` - figures in `figures/`

Step 3 takes about half an hour with 14 workers, mostly because of SA. Results are already included (`results_budget_*.parquet`, `convergence_budget_*.parquet`).

## Files

- `problem.py` - problem definition (encoding, constraints, objectives)
- `metaheuristics.py` - random search, GA, NSGA-II, SA
- `relaxation.py` - relaxation procedure (bounds first, then features from reluctancy list)
- `users.py` - simulated user preferences, with and without the SHAP nudge
- `experiments.py` - runs everything

Run on Python 3.14 with scikit-learn 1.8.0.
