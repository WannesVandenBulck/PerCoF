# PerCoF: Personalised Counterfactuals with Metaheuristics

PerCoF finds counterfactual explanations that respect what an individual applicant is willing and able to change. When no counterfactual fits their constraints, it returns the smallest relaxation of those constraints that does.

This repository compares three metaheuristics (a genetic algorithm, NSGA-II and simulated annealing) on this problem, using the German Credit dataset.

## Status

- [x] Data preparation and predictive model
- [ ] PerCoF problem definition (encoding, constraints, objectives)
- [ ] Metaheuristics (GA, NSGA-II, SA)
- [ ] Relaxation procedure
- [ ] Experiments and evaluation

## Data and model

- **Dataset:** Statlog German Credit (UCI, ID 144): 1,000 applicants, 20 features. Target: 1 = bad credit risk, 0 = good.
- **Protected attributes** (age, sex, foreign worker) are excluded from the model.
- **Model:** random forest (default settings, balanced class weights) on a stratified 80/20 split.
  - Accuracy: 79.00%
  - AUC: 84.14%
- **Rejected applicants:** 44 of the 200 test applicants (predicted probability of bad credit ≥ 0.5). These are the applicants PerCoF explains counterfatuals for.

## Actionable features

Only these features can be selected by a user; all others stay fixed.

Numeric: duration, amount, installment_rate, number_credits
Ordinal: status, savings, job
Nominal: other_debtors, property, other_installment_plans, housing

Meaningless target values are excluded: a counterfactual never moves to "no checking account", "unknown/no savings" or "unknown/no property".

## Contents of `percof_setup.pkl`

| Key | Contents |
|---|---|
| `feature_spec` | Type, order, allowed targets and actionability per feature |
| `model_features` | Column order expected by the model |
| `stats` | Min, max and MAD per numeric feature (training data) |
| `rejected` | The 44 rejected test applicants |
| `feature_mappings` | Readable labels for encoded values |
| `feature_desc` | Readable feature descriptions |

## Usage

```bash
pip install pandas numpy scikit-learn ucimlrepo pyarrow
```

Run `prep_credit.ipynb` from top to bottom to regenerate the model and setup files. Load them in Python with:

```python
import pickle

with open("RF.pkl", "rb") as f:
    model = pickle.load(f)
with open("percof_setup.pkl", "rb") as f:
    setup = pickle.load(f)
```

## Data source

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
