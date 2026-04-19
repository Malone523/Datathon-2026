"""
Scrap Rate Minimisation — Full Analysis Pipeline
=================================================
Methods used:
  1. Ordinary Least Squares (OLS) regression        — statsmodels
  2. Gradient Boosting Machine (GBM)                — scikit-learn
  3. Partial Dependence Analysis                    — scikit-learn
  4. Differential Evolution global optimiser        — scipy

Dependencies:
    pip install pandas numpy statsmodels scikit-learn scipy
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv("synthetic_injection_molding_demo.csv")

TARGET = "scrap_rate_pct"

# Columns to exclude: targets, counters, derived outputs
EXCLUDE = {
    "scrap_rate_pct", "scrap_count", "pass_fail_flag",
    "parts_produced", "energy_kwh_interval",
    "cycle_time_s", "cavity_count", "part_weight_g",
}

# All numeric columns not excluded become candidate predictors
ALL_FEATURES = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in EXCLUDE
]

# Split into controllable (process knobs) vs uncontrollable (environment)
UNCONTROLLABLE = {"ambient_temperature_c", "ambient_humidity_pct", "operator_experience_level"}
CONTROLLABLE   = [c for c in ALL_FEATURES if c not in UNCONTROLLABLE]

# Drop rows with any missing value in the columns we use
data = df[ALL_FEATURES + [TARGET]].dropna()

print(f"Dataset: {len(data)} rows, {len(ALL_FEATURES)} features")
print(f"Target mean: {data[TARGET].mean():.4f}%  std: {data[TARGET].std():.4f}%\n")


# ─────────────────────────────────────────────
# 2. OLS REGRESSION
#    Estimates a linear model:
#      scrap = β₀ + Σ βᵢ·xᵢ + ε
#    Solved analytically via:
#      β = (XᵀX)⁻¹ Xᵀy
#
#    statsmodels.OLS uses this exact closed-form
#    solution and also computes standard errors,
#    t-statistics, and p-values for each coefficient.
# ─────────────────────────────────────────────

X_ols = sm.add_constant(data[ALL_FEATURES])   # prepends intercept column
y     = data[TARGET]

ols_model = sm.OLS(y, X_ols).fit()

print("=" * 60)
print("OLS REGRESSION RESULTS")
print("=" * 60)
print(f"R²          : {ols_model.rsquared:.4f}")
print(f"Adjusted R² : {ols_model.rsquared_adj:.4f}")
print(f"n           : {len(data)}\n")

print(f"{'Variable':<35} {'Coef':>10} {'Std Eff':>10} {'p-val':>10} {'Sig':>5}")
print("-" * 72)
for col in ALL_FEATURES:
    coef     = ols_model.params[col]
    p_val    = ols_model.pvalues[col]
    std_eff  = coef * data[col].std()           # standardised effect = β × σ
    sig      = ("***" if p_val < 0.001 else
                "**"  if p_val < 0.01  else
                "*"   if p_val < 0.05  else "ns")
    print(f"  {col:<33} {coef:>+10.5f} {std_eff:>+10.4f} {p_val:>10.4f} {sig:>5}")

print("\nIntercept :", round(ols_model.params["const"], 4))
print()


# ─────────────────────────────────────────────
# 3. GRADIENT BOOSTING MACHINE (GBM)
#    Builds an ensemble of decision trees, where
#    each tree corrects the residuals of the
#    previous one. Captures nonlinear effects
#    and interaction terms that OLS misses.
#
#    Hyperparameters used:
#      n_estimators = 400   (number of trees)
#      max_depth    = 4     (tree depth)
#      learning_rate= 0.05  (shrinkage)
#      subsample    = 0.8   (stochastic sampling)
#
#    Feature importance = total reduction in MSE
#    attributed to each feature across all trees.
# ─────────────────────────────────────────────

gbm = GradientBoostingRegressor(
    n_estimators  = 400,
    max_depth     = 4,
    learning_rate = 0.05,
    subsample     = 0.8,
    random_state  = 42,
)
gbm.fit(data[ALL_FEATURES], y)

cv_scores = cross_val_score(
    gbm, data[ALL_FEATURES], y,
    cv=5, scoring="r2"
)

print("=" * 60)
print("GRADIENT BOOSTING RESULTS")
print("=" * 60)
print(f"Train R²   : {gbm.score(data[ALL_FEATURES], y):.4f}")
print(f"CV R² (5×) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

importances = sorted(
    zip(ALL_FEATURES, gbm.feature_importances_),
    key=lambda x: -x[1]
)
print(f"{'Feature':<35} {'Importance':>12}")
print("-" * 50)
for feat, imp in importances:
    bar = "█" * int(imp * 200)
    print(f"  {feat:<33} {imp:>10.4f}  {bar}")
print()


# ─────────────────────────────────────────────
# 4. PARTIAL DEPENDENCE ANALYSIS
#    For each top feature, sweeps its value
#    across its observed range while holding all
#    other features at their training mean.
#    This reveals the marginal (nonlinear) shape
#    of each variable's effect on scrap rate.
# ─────────────────────────────────────────────

TOP_FEATURES = [f for f, _ in importances[:6]]
mean_vals    = {c: float(data[c].mean()) for c in ALL_FEATURES}

print("=" * 60)
print("PARTIAL DEPENDENCE OPTIMA (top 6 GBM features)")
print("=" * 60)
print(f"{'Feature':<35} {'Current mean':>14} {'PD optimal':>12} {'Min scrap':>12}")
print("-" * 76)

for feat in TOP_FEATURES:
    lo   = float(data[feat].quantile(0.02))
    hi   = float(data[feat].quantile(0.98))
    grid = np.linspace(lo, hi, 200)

    pd_preds = []
    for v in grid:
        row         = {c: mean_vals[c] for c in ALL_FEATURES}
        row[feat]   = v
        X_row       = np.array([[row[c] for c in ALL_FEATURES]])
        pd_preds.append(float(gbm.predict(X_row)[0]))

    best_idx = int(np.argmin(pd_preds))
    print(f"  {feat:<33} {mean_vals[feat]:>14.3f} "
          f"{grid[best_idx]:>12.3f} {pd_preds[best_idx]:>11.4f}%")
print()


# ─────────────────────────────────────────────
# 5. GLOBAL OPTIMISATION — DIFFERENTIAL EVOLUTION
#    Finds the combination of controllable
#    variable settings that minimises predicted
#    scrap rate according to the GBM model.
#
#    Differential Evolution is a population-based
#    stochastic global optimiser. It does not
#    require gradients and handles non-convex,
#    multimodal search spaces well.
#
#    Algorithm (SciPy default — Storn & Price 1997):
#      1. Initialise population of N candidate
#         solutions, each a vector of variable
#         settings drawn uniformly from bounds.
#      2. For each candidate x:
#           a. Pick 3 distinct others: a, b, c
#           b. Mutant v = a + F·(b − c)
#              (F = mutation factor, here 0.5–1.5)
#           c. Trial u = crossover of x and v
#              (crossover prob = 0.7)
#           d. If GBM(u) < GBM(x): replace x with u
#      3. Repeat until convergence (tol=1e-9)
#         or max iterations (1000) reached.
#
#    Uncontrollable variables are fixed at their
#    observed mean throughout. Bounds for each
#    controllable variable are set to the 2nd–98th
#    percentile of observed values.
# ─────────────────────────────────────────────

# Bounds: 2nd–98th percentile of observed data
bounds = [
    (float(data[c].quantile(0.02)), float(data[c].quantile(0.98)))
    for c in CONTROLLABLE
]

def predict_scrap(x_controllable):
    """Objective function: GBM prediction with uncontrollable vars at mean."""
    row = {c: mean_vals[c] for c in ALL_FEATURES}
    for i, col in enumerate(CONTROLLABLE):
        row[col] = x_controllable[i]
    X = np.array([[row[c] for c in ALL_FEATURES]])
    return float(gbm.predict(X)[0])

# Baseline: all controllable vars at their mean
baseline_input  = [mean_vals[c] for c in CONTROLLABLE]
baseline_scrap  = predict_scrap(baseline_input)

result = differential_evolution(
    predict_scrap,
    bounds,
    seed        = 42,
    maxiter     = 1000,
    tol         = 1e-9,
    popsize     = 20,
    mutation    = (0.5, 1.5),
    recombination = 0.7,
)

opt_scrap = result.fun

print("=" * 60)
print("OPTIMISATION RESULTS (Differential Evolution + GBM)")
print("=" * 60)
print(f"Baseline scrap (vars at mean) : {baseline_scrap:.4f}%")
print(f"Actual dataset mean scrap     : {data[TARGET].mean():.4f}%")
print(f"Optimised scrap               : {opt_scrap:.4f}%")
print(f"Absolute reduction            : {baseline_scrap - opt_scrap:.4f} pp")
print(f"Relative reduction            : {(baseline_scrap - opt_scrap) / baseline_scrap * 100:.1f}%")
print(f"Converged                     : {result.success}  ({result.message})\n")


# ─────────────────────────────────────────────
# 6. REPORT OPTIMAL SETTINGS
#    For each controllable variable, show:
#      - current mean in dataset
#      - optimal value found by DE
#      - direction of change
#      - marginal OLS contribution at that delta
# ─────────────────────────────────────────────

print("=" * 60)
print("OPTIMAL SETTINGS — RANKED BY MARGINAL OLS IMPACT")
print("=" * 60)
print(f"{'Variable':<35} {'Mean':>8} {'Optimal':>10} {'Δ':>10} {'OLS Δscrap':>12}")
print("-" * 78)

changes = []
for i, col in enumerate(CONTROLLABLE):
    old    = mean_vals[col]
    new    = result.x[i]
    delta  = new - old
    ols_impact = (ols_model.params.get(col, 0.0) * delta
                  if col in ols_model.params else 0.0)
    changes.append((abs(ols_impact), col, old, new, delta, ols_impact))

changes.sort(reverse=True)
for _, col, old, new, delta, impact in changes:
    print(f"  {col:<33} {old:>8.3f} {new:>10.3f} {delta:>+10.3f} {impact:>+11.4f}%")


# ─────────────────────────────────────────────
# 7. OLS FORMULA (significant terms only)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("OLS FORMULA  (p < 0.05 terms only)")
print("=" * 60)
sig_terms = [
    (col, ols_model.params[col], ols_model.pvalues[col])
    for col in ALL_FEATURES
    if ols_model.pvalues[col] < 0.05
]
sig_terms.sort(key=lambda x: abs(x[1] * data[x[0]].std()), reverse=True)

print(f"  scrap_rate_pct = {ols_model.params['const']:.4f}")
for col, coef, pval in sig_terms:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
    print(f"    {'+' if coef >= 0 else ''}{coef:.6f} × {col}  [{sig}]")

print(f"\n  R² = {ols_model.rsquared:.4f}")
print(f"  GBM cross-validated R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  n = {len(data)}")