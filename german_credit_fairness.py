"""
CS 6603 - AI, Ethics, and Society
Homework Project #5: Fairness and Bias  -  German Credit Data Set

HOW TO USE / REVIEW THIS FILE
-----------------------------
* Every choice the assignment leaves open is collected in the CONFIG section
  directly below. Each option is marked  # >>> EDIT  and the alternatives are
  listed in the comment next to it. You should not need to touch anything below
  the CONFIG section to try a different choice.
* The file is split into "# %%" cells, one per assignment step, so it can be run
  top-to-bottom as a script (`python german_credit_fairness.py`) or cell-by-cell
  in VS Code / Jupyter / Spyder.
* All charts are saved as PNG and all tables as CSV in OUTPUT_DIR, and a plain-text
  summary of every number the report asks for is written to OUTPUT_DIR/results_summary.txt.

Requirements: python >= 3.9, numpy, pandas, scikit-learn, matplotlib
(no AIF360 install needed - the fairness metrics are implemented below using the
same definitions AIF360 / What-If Tool use, see Step 5).

Data files needed (from the UCI "german" folder): german.data
(german.data-numeric and german.doc are not read by the code; german.doc was used
to write the column names and codes below.)
"""

# %% =============================================================================
# CONFIG  -  all editable choices live here
# ===============================================================================
import argparse
from pathlib import Path

# --- Files -----------------------------------------------------------------------
DATA_PATH = Path("data/german.data")            # >>> EDIT path to german.data (or pass --data on the command line)
OUTPUT_DIR = Path("outputs")               # >>> EDIT folder for figures / tables

# --- Reproducibility -------------------------------------------------------------
RANDOM_SEED = 27                           # >>> EDIT any integer; changes the random train/test split
TEST_SIZE = 0.5                            # assignment requires equally-sized train/test sets (0.5)
STRATIFY_SPLIT = False                     # >>> EDIT True = keep same good/bad ratio in both halves

# --- Step 3.2: creditworthiness formula ------------------------------------------
# "logistic" : logistic regression fitted on the TRAINING set;
#              score = 100 * P(good credit risk). Formula/coefficients are exported.
# "points"   : a hand-written, fully transparent points scorecard (see POINTS_TABLE
#              in Step 3.2); score rescaled to 0-100. No fitting needed.
SCORE_METHOD = "logistic"                  # >>> EDIT "logistic" or "points"
LOGISTIC_C = 1.0                           # >>> EDIT regularisation strength (smaller = stronger)
ROUND_SCORES = True                        # >>> EDIT True = integer scores 0..100 (cleaner histograms)

# Variables the score is NOT allowed to use (protected-class information).
# The assignment forbids age, gender, national origin, etc. "personal_status_sex"
# mixes sex and marital status (both protected under ECOA), "foreign_worker" is a
# proxy for national origin.
EXCLUDED_FROM_SCORE = ["age", "personal_status_sex", "foreign_worker"]   # >>> EDIT (only add, never remove protected ones)

# --- Step 3.3 / 3.4: protected attribute and groups -------------------------------
# "age"            -> split at AGE_CUTOFF
# "sex"            -> derived from attribute 9 (A92, A95 = female, else male)
# "foreign_worker" -> attribute 20 (A201 = foreign worker)
PROTECTED_ATTRIBUTE = "age"                # >>> EDIT "age", "sex" or "foreign_worker"

AGE_CUTOFF = 25                            # >>> EDIT e.g. 25 (Kamiran & Calders / AIF360 use 25) or 40 (assignment example)
UNPRIVILEGED_AGE_GROUP = "young"           # >>> EDIT "young" (age <= cutoff is unprivileged)
                                           #          or "old"  (age >  cutoff is unprivileged)
UNPRIVILEGED_SEX = "female"                # >>> EDIT "female" or "male"   (only used if PROTECTED_ATTRIBUTE == "sex")
UNPRIVILEGED_FOREIGN = "foreign"           # >>> EDIT "foreign" or "domestic" (only used if PROTECTED_ATTRIBUTE == "foreign_worker")

# --- Step 4: profit table (from assignment Table 2) -------------------------------
PROFIT_TP = 10    # dataset approves, we approve
PROFIT_FN = -5    # dataset approves, we deny
PROFIT_FP = -10   # dataset denies,   we approve
PROFIT_TN = 0     # dataset denies,   we deny

THRESHOLD_RANGE = range(1, 101)            # thresholds tried (assignment: a number between 1 and 100)
# The assignment says approving everyone (threshold 0) is not a valid answer. With
# this flag on, any threshold that approves 100% of the people it applies to is
# treated as "approve-all" and is not allowed.
FORBID_APPROVE_ALL = True                  # >>> EDIT

# --- Step 5: fairness metrics -----------------------------------------------------
# Choose TWO of: "disparate_impact", "statistical_parity_difference",
#                "equal_opportunity_difference", "average_odds_difference"
FAIRNESS_METRICS = ["disparate_impact", "statistical_parity_difference"]   # >>> EDIT

# --- Step 6: bias mitigation ------------------------------------------------------
MITIGATION_METRIC = "disparate_impact"     # >>> EDIT must be one of FAIRNESS_METRICS
INITIAL_THRESHOLDS = None                  # >>> EDIT None = start both groups at the Step-4 threshold,
                                           #          or a tuple (privileged, unprivileged), e.g. (50, 40)
# How "minimise bias while maximising profit" is turned into one choice:
# "constrained" : highest profit among threshold pairs whose metric is within
#                 FAIRNESS_TOLERANCE of the ideal value (1 for DI, 0 for differences).
#                 If no pair qualifies, the least-biased pair is used.
# "weighted"    : maximise  profit/|max profit|  -  BIAS_WEIGHT * |metric - ideal|
OPTIMIZATION_MODE = "constrained"          # >>> EDIT "constrained" or "weighted"
FAIRNESS_TOLERANCE = 0.02                  # >>> EDIT used by "constrained"
BIAS_WEIGHT = 5.0                          # >>> EDIT used by "weighted"

# --- Extras -----------------------------------------------------------------------
EVALUATE_ON_TEST = True                    # >>> EDIT also report results on the held-out test set
HIST_BIN_WIDTH = 5                         # >>> EDIT histogram bar width in score points
SHOW_PLOTS = False                         # >>> EDIT True = also pop up each figure window

# %% =============================================================================
# Imports, plotting style
# ===============================================================================
import numpy as np
import pandas as pd
import matplotlib
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Colours (colour-blind-safe pair) - edit here to restyle every chart.
C_GOOD = "#628cb6"        # good credit risk / approved / privileged
C_BAD = "#c97b63"         # bad credit risk / denied / unprivileged
C_LINE = "#4d4d4d"        # threshold lines
C_FAIR = "#cfcfcf"        # "fair" band in metric charts
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

# Command-line override for the data path / output folder (optional)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data", type=Path, default=None)
_parser.add_argument("--out", type=Path, default=None)
_args, _ = _parser.parse_known_args()
if _args.data is not None:
    DATA_PATH = _args.data
if _args.out is not None:
    OUTPUT_DIR = _args.out
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_LINES = []


def report(text=""):
    """Print and remember a line for results_summary.txt."""
    print(text)
    SUMMARY_LINES.append(str(text))


def save_table(df, name, index=True):
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    report(df.to_string(index=index))
    report(f"  [table saved -> {path}]")


def save_fig(fig, name):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    report(f"  [figure saved -> {path}]")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def header(title):
    report("")
    report("=" * 78)
    report(title)
    report("=" * 78)


# %% =============================================================================
# STEP 1 - Dataset selection
# ===============================================================================
header("STEP 1 - Dataset selection")
report("Dataset: Statlog German Credit Data (Prof. Hans Hofmann, Univ. Hamburg), file german.data")

# Column names / codes taken from german.doc
COLUMNS = [
    "checking_status",         # A1  qualitative
    "duration_months",         # A2  numerical
    "credit_history",          # A3  qualitative
    "purpose",                 # A4  qualitative
    "credit_amount",           # A5  numerical
    "savings",                 # A6  qualitative
    "employment_since",        # A7  qualitative
    "installment_rate",        # A8  numerical
    "personal_status_sex",     # A9  qualitative (sex + marital status)
    "other_debtors",           # A10 qualitative
    "residence_since",         # A11 numerical
    "property",                # A12 qualitative
    "age",                     # A13 numerical
    "other_installment_plans", # A14 qualitative
    "housing",                 # A15 qualitative
    "existing_credits",        # A16 numerical
    "job",                     # A17 qualitative
    "num_dependents",          # A18 numerical
    "telephone",               # A19 qualitative
    "foreign_worker",          # A20 qualitative
    "class",                   # outcome: 1 = good, 2 = bad
]
NUMERIC_COLS = ["duration_months", "credit_amount", "installment_rate", "residence_since",
                "age", "existing_credits", "num_dependents"]

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)

# Helper columns (NOT used by the score - only for grouping)
df["sex"] = np.where(df["personal_status_sex"].isin(["A92", "A95"]), "female", "male")
df["foreign"] = np.where(df["foreign_worker"] == "A201", "foreign", "domestic")
df["approved_actual"] = (df["class"] == 1).astype(int)   # 1 = dataset approved (good credit risk)

# %% =============================================================================
# STEP 2 - Explore the dataset
# ===============================================================================
header("STEP 2 - Explore the dataset")
n_obs = len(df)
n_vars = len(COLUMNS)
report(f"2.1 Number of observations: {n_obs}")
report(f"2.2 Number of variables:    {n_vars} (20 attributes + 1 outcome variable 'class'; "
       f"7 numerical, 13 categorical attributes)")
report("2.3 Variables tied to a legally recognised protected class:")
protected_table = pd.DataFrame([
    ["age (Attribute 13)", "Age",
     "US: Equal Credit Opportunity Act (ECOA, 1974; 15 U.S.C. 1691) / Regulation B; "
     "Age Discrimination Act of 1975. Germany: General Equal Treatment Act (AGG, 2006)."],
    ["personal_status_sex (Attribute 9)", "Sex and marital status",
     "US: ECOA (sex and marital status), Fair Housing Act (sex). "
     "Germany: AGG (sex); EU Gender Goods & Services Directive 2004/113/EC."],
    ["foreign_worker (Attribute 20)", "National origin / citizenship (proxy)",
     "US: ECOA (national origin), Civil Rights Act of 1964 Title VI, Fair Housing Act. "
     "Germany: AGG (ethnic origin); EU Race Equality Directive 2000/43/EC."],
], columns=["Variable", "Protected class", "Legal precedent / law"])
save_table(protected_table, "step2_protected_variables", index=False)

report("\nOther quick facts:")
report(f"  Outcome distribution: {df['class'].map({1: 'good (1)', 2: 'bad (2)'}).value_counts().to_dict()}")
report(f"  Age range: {df['age'].min()}-{df['age'].max()}, median {df['age'].median():.0f}")
report(f"  Missing values: {int(df[COLUMNS].isna().sum().sum())}")

# %% =============================================================================
# STEP 3.1 - Outcome variable
# ===============================================================================
header("STEP 3.1 - Outcome variable")
report("Outcome variable: 'class'  (1 = Good credit risk -> loan approved, "
       "2 = Bad credit risk -> loan denied)")

# %% =============================================================================
# STEP 3.3 / 3.4 - Protected attribute, privileged / unprivileged groups
# (done before 3.2 in code only because the split in 3.5 needs the group column)
# ===============================================================================
header("STEP 3.3 / 3.4 - Protected attribute and groups")
if PROTECTED_ATTRIBUTE == "age":
    young = df["age"] <= AGE_CUTOFF
    unpriv_mask = young if UNPRIVILEGED_AGE_GROUP == "young" else ~young
    UNPRIV_LABEL = f"Young (age <= {AGE_CUTOFF})" if UNPRIVILEGED_AGE_GROUP == "young" else f"Old (age > {AGE_CUTOFF})"
    PRIV_LABEL = f"Old (age > {AGE_CUTOFF})" if UNPRIVILEGED_AGE_GROUP == "young" else f"Young (age <= {AGE_CUTOFF})"
elif PROTECTED_ATTRIBUTE == "sex":
    unpriv_mask = df["sex"] == UNPRIVILEGED_SEX
    UNPRIV_LABEL = UNPRIVILEGED_SEX.capitalize()
    PRIV_LABEL = "Male" if UNPRIVILEGED_SEX == "female" else "Female"
elif PROTECTED_ATTRIBUTE == "foreign_worker":
    unpriv_mask = df["foreign"] == UNPRIVILEGED_FOREIGN
    UNPRIV_LABEL = f"{UNPRIVILEGED_FOREIGN.capitalize()} worker"
    PRIV_LABEL = "Domestic worker" if UNPRIVILEGED_FOREIGN == "foreign" else "Foreign worker"
else:
    raise ValueError("PROTECTED_ATTRIBUTE must be 'age', 'sex' or 'foreign_worker'")

df["privileged"] = (~unpriv_mask).astype(int)   # 1 = privileged, 0 = unprivileged
report(f"Protected class attribute: {PROTECTED_ATTRIBUTE}")
report(f"Privileged group:   {PRIV_LABEL}")
report(f"Unprivileged group: {UNPRIV_LABEL}")

# %% =============================================================================
# STEP 3.5 - Split into train / test (50 / 50)
# ===============================================================================
header("STEP 3.5 - Train / test split")
train, test = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    stratify=df["class"] if STRATIFY_SPLIT else None,
)
train, test = train.copy(), test.copy()
split_counts = pd.DataFrame({
    "Training set": [int((train.privileged == 1).sum()), int((train.privileged == 0).sum()), len(train)],
    "Testing set": [int((test.privileged == 1).sum()), int((test.privileged == 0).sum()), len(test)],
}, index=[f"Privileged: {PRIV_LABEL}", f"Unprivileged: {UNPRIV_LABEL}", "Total"])
save_table(split_counts, "step3_5_group_counts")

# %% =============================================================================
# STEP 3.2 - Creditworthiness formula (0-100), protected attributes excluded
# ===============================================================================
header("STEP 3.2 - Creditworthiness formula")
SCORE_FEATURES = [c for c in COLUMNS if c not in EXCLUDED_FROM_SCORE + ["class"]]
SCORE_NUMERIC = [c for c in SCORE_FEATURES if c in NUMERIC_COLS]
SCORE_CATEG = [c for c in SCORE_FEATURES if c not in NUMERIC_COLS]
report(f"Variables used ({len(SCORE_FEATURES)}): {SCORE_FEATURES}")
report(f"Variables excluded (protected): {EXCLUDED_FROM_SCORE}")

# ---- Option B: hand-written points scorecard (used if SCORE_METHOD == "points") ----
# >>> EDIT points freely. Higher = more creditworthy. Raw total is rescaled to 0-100.
POINTS_TABLE = {
    "checking_status":   {"A11": 0, "A12": 8, "A13": 15, "A14": 20},
    "credit_history":    {"A30": 0, "A31": 2, "A32": 8, "A33": 8, "A34": 14},
    "savings":           {"A61": 0, "A62": 3, "A63": 6, "A64": 10, "A65": 8},
    "employment_since":  {"A71": 0, "A72": 1, "A73": 4, "A74": 7, "A75": 6},
    "other_installment_plans": {"A141": 0, "A142": 0, "A143": 6},
    "housing":           {"A151": 0, "A152": 5, "A153": 2},
    "property":          {"A121": 6, "A122": 4, "A123": 3, "A124": 0},
}
POINTS_DURATION_MAX = 15        # points for duration: max(0, 15 - duration_months / 4)
POINTS_INSTALLMENT_MAX = 6      # points for installment rate: 2 * (4 - installment_rate)


def points_raw(data):
    total = np.zeros(len(data))
    for col, mapping in POINTS_TABLE.items():
        total += data[col].map(mapping).fillna(0).to_numpy()
    total += np.clip(POINTS_DURATION_MAX - data["duration_months"].to_numpy() / 4, 0, None)
    total += 2 * (4 - data["installment_rate"].to_numpy())
    return total


POINTS_MIN = 0.0
POINTS_MAX = (sum(max(m.values()) for m in POINTS_TABLE.values())
              + POINTS_DURATION_MAX + POINTS_INSTALLMENT_MAX)

# ---- Option A: logistic regression (default) ----
logit_model = None
if SCORE_METHOD == "logistic":
    logit_model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), SCORE_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), SCORE_CATEG),
        ])),
        ("clf", LogisticRegression(C=LOGISTIC_C, max_iter=5000)),
    ])
    # Fitted on the TRAINING set only (target = 1 if good credit risk)
    logit_model.fit(train[SCORE_FEATURES], train["approved_actual"])

    # Convert to an explicit formula in ORIGINAL units (undo standardisation) so the
    # report can state it:  score = 100 / (1 + exp(-(b0 + sum b_i * x_i)))
    prep = logit_model.named_steps["prep"]
    clf = logit_model.named_steps["clf"]
    scaler = prep.named_transformers_["num"]
    ohe = prep.named_transformers_["cat"]
    coefs = clf.coef_[0]
    n_num = len(SCORE_NUMERIC)
    num_coef = coefs[:n_num] / scaler.scale_
    intercept = clf.intercept_[0] - np.sum(coefs[:n_num] * scaler.mean_ / scaler.scale_)
    rows = [("intercept", "", intercept)]
    rows += [(c, "per unit", b) for c, b in zip(SCORE_NUMERIC, num_coef)]
    for name, b in zip(ohe.get_feature_names_out(SCORE_CATEG), coefs[n_num:]):
        var, level = name.rsplit("_", 1)
        rows.append((var, f"= {level} (vs. baseline)", b))
    formula_df = pd.DataFrame(rows, columns=["variable", "term", "coefficient"])
    report("Formula: Creditworthiness = 100 / (1 + exp(-z)),  z = b0 + SUM(b_i * x_i)")
    report("  (= 100 x predicted probability of being a Good Credit Risk; "
           "categorical x_i are 0/1 indicators; baseline level = first code of each variable)")
    save_table(formula_df.round(4), "step3_2_logistic_formula_coefficients", index=False)
elif SCORE_METHOD == "points":
    report("Formula: Creditworthiness = 100 * (points - min) / (max - min), where points =")
    for col, mapping in POINTS_TABLE.items():
        report(f"  {col}: {mapping}")
    report(f"  + max(0, {POINTS_DURATION_MAX} - duration_months/4)")
    report("  + 2 * (4 - installment_rate)")
    report(f"  (min = {POINTS_MIN}, max = {POINTS_MAX})")
else:
    raise ValueError("SCORE_METHOD must be 'logistic' or 'points'")


def creditworthiness(data):
    """Return creditworthiness scores (0-100) for a DataFrame of applicants."""
    if SCORE_METHOD == "logistic":
        s = 100 * logit_model.predict_proba(data[SCORE_FEATURES])[:, 1]
    else:
        s = 100 * (points_raw(data) - POINTS_MIN) / (POINTS_MAX - POINTS_MIN)
    s = np.clip(s, 0, 100)
    return np.round(s).astype(int) if ROUND_SCORES else s


train["score"] = creditworthiness(train)
test["score"] = creditworthiness(test)
report(f"\nTraining-set score summary: {train['score'].describe().round(1).to_dict()}")

# %% =============================================================================
# Shared helpers: profit and fairness metrics
# ===============================================================================


def confusion(y_true, approve):
    """Counts with 'positive' = approved / good credit."""
    y_true = np.asarray(y_true).astype(bool)
    approve = np.asarray(approve).astype(bool)
    return dict(TP=int(np.sum(y_true & approve)), FN=int(np.sum(y_true & ~approve)),
                FP=int(np.sum(~y_true & approve)), TN=int(np.sum(~y_true & ~approve)))


def profit_from_counts(c):
    return c["TP"] * PROFIT_TP + c["FN"] * PROFIT_FN + c["FP"] * PROFIT_FP + c["TN"] * PROFIT_TN


def total_profit(data, approve):
    return profit_from_counts(confusion(data["approved_actual"], approve))


def group_rates(data, approve):
    """Favourable rate, TPR and FPR for the privileged (1) and unprivileged (0) groups."""
    approve = np.asarray(approve).astype(bool)
    out = {}
    for g in (1, 0):
        m = (data["privileged"] == g).to_numpy()
        c = confusion(data["approved_actual"].to_numpy()[m], approve[m])
        n = m.sum()
        out[g] = dict(
            n=n,
            fav=(c["TP"] + c["FP"]) / n if n else np.nan,
            tpr=c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else np.nan,
            fpr=c["FP"] / (c["FP"] + c["TN"]) if (c["FP"] + c["TN"]) else np.nan,
        )
    return out


def _safe_div(a, b):
    """a / b that returns NaN when b == 0 (works for numbers and numpy arrays)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)
    return float(out) if out.ndim == 0 else out


# Metric definitions (identical to AIF360's ClassificationMetric / What-If Tool)
# Each entry: (display name, function(rates), ideal value, (fair_low, fair_high), formula text)
METRICS = {
    "disparate_impact": (
        "Disparate Impact",
        lambda r: _safe_div(r[0]["fav"], r[1]["fav"]),
        1.0, (0.8, 1.25),
        "P(approved | unpriv) / P(approved | priv)"),
    "statistical_parity_difference": (
        "Statistical Parity Difference",
        lambda r: r[0]["fav"] - r[1]["fav"],
        0.0, (-0.1, 0.1),
        "P(approved | unpriv) - P(approved | priv)"),
    "equal_opportunity_difference": (
        "Equal Opportunity Difference",
        lambda r: r[0]["tpr"] - r[1]["tpr"],
        0.0, (-0.1, 0.1),
        "TPR(unpriv) - TPR(priv)"),
    "average_odds_difference": (
        "Average Odds Difference",
        lambda r: 0.5 * ((r[0]["fpr"] - r[1]["fpr"]) + (r[0]["tpr"] - r[1]["tpr"])),
        0.0, (-0.1, 0.1),
        "0.5 * [(FPR_u - FPR_p) + (TPR_u - TPR_p)]"),
}
assert len(FAIRNESS_METRICS) == 2 and all(m in METRICS for m in FAIRNESS_METRICS), \
    "FAIRNESS_METRICS must list exactly two keys of METRICS"
assert MITIGATION_METRIC in FAIRNESS_METRICS, "MITIGATION_METRIC must be one of FAIRNESS_METRICS"


def compute_metrics(data, approve):
    r = group_rates(data, approve)
    return {k: METRICS[k][1](r) for k in METRICS}, r


def bias_verdict(key, value):
    name, _, ideal, (lo, hi), _ = METRICS[key]
    if np.isnan(value):
        return "undefined"
    if lo <= value <= hi:
        return "No significant bias (within fair range)"
    favoured = "privileged" if value < ideal else "unprivileged"
    return f"BIAS - favours the {favoured} group"


def favorable_table(data, approve):
    approve = np.asarray(approve).astype(bool)
    rows = {}
    for g, label in ((1, f"Privileged: {PRIV_LABEL}"), (0, f"Unprivileged: {UNPRIV_LABEL}")):
        m = (data["privileged"] == g).to_numpy()
        fav, n = int(approve[m].sum()), int(m.sum())
        rows[label] = {"Favorable (Approved)": fav, "Unfavorable (Denied)": n - fav,
                       "Total": n, "Approval rate": round(fav / n, 3) if n else np.nan}
    return pd.DataFrame(rows).T.astype({"Favorable (Approved)": int, "Unfavorable (Denied)": int, "Total": int})


def is_approve_all(scores, t):
    return bool(np.all(np.asarray(scores) >= t))


# %% =============================================================================
# STEP 4.1 - Histogram of creditworthiness (training set)
# ===============================================================================
header("STEP 4.1 - Histogram of creditworthiness (training set)")
BINS = np.arange(0, 100 + HIST_BIN_WIDTH, HIST_BIN_WIDTH)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(train["score"], bins=BINS, color=C_GOOD, edgecolor="white", linewidth=1.5)
ax.set_xlabel("Creditworthiness (0-100)")
ax.set_ylabel("Number of customers")
ax.set_title("Creditworthiness of training-set customers")
ax.set_xticks(range(0, 101, 10))
save_fig(fig, "step4_1_histogram")

# %% =============================================================================
# STEP 4.2 - Threshold that maximises profit (training set)
# ===============================================================================
header("STEP 4.2 - Profit-maximising threshold (training set)")
profit_curve = []
for t in THRESHOLD_RANGE:
    approve = train["score"] >= t
    allowed = not (FORBID_APPROVE_ALL and is_approve_all(train["score"], t))
    profit_curve.append({"threshold": t, "profit": total_profit(train, approve),
                         "approval_rate": approve.mean(), "allowed": allowed})
profit_curve = pd.DataFrame(profit_curve)
profit_curve.to_csv(OUTPUT_DIR / "step4_2_profit_by_threshold.csv", index=False)
valid = profit_curve[profit_curve["allowed"]]
best_profit = valid["profit"].max()
ties = valid.loc[valid["profit"] == best_profit, "threshold"].tolist()
BEST_T = int(ties[0])   # >>> if several thresholds tie, the lowest one is used
report(f"Threshold X that maximises profit: {BEST_T}" + (f"  (tied: {ties})" if len(ties) > 1 else ""))
report(f"Maximum profit on training set:   {best_profit}")
c = confusion(train["approved_actual"], train["score"] >= BEST_T)
report(f"  Confusion counts at X: {c}  ->  {c['TP']}*{PROFIT_TP} + {c['FN']}*({PROFIT_FN}) "
       f"+ {c['FP']}*({PROFIT_FP}) + {c['TN']}*{PROFIT_TN} = {profit_from_counts(c)}")
report(f"  Approval rate at X: {(train['score'] >= BEST_T).mean():.1%}")
report(f"  (Reference: dataset itself approves {train['approved_actual'].mean():.1%}; "
       f"approving everyone would earn {total_profit(train, np.ones(len(train), bool))})")

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(profit_curve["threshold"], profit_curve["profit"], color=C_GOOD, linewidth=2)
ax.axvline(BEST_T, color=C_LINE, linestyle="--", linewidth=1.2)
ax.annotate(f"X = {BEST_T}\nprofit = {best_profit}", xy=(BEST_T, best_profit),
            xytext=(8, -30), textcoords="offset points", fontsize=9)
ax.set_xlabel("Threshold (approve if creditworthiness >= threshold)")
ax.set_ylabel("Profit (training set)")
ax.set_title("Profit vs. approval threshold")
save_fig(fig, "step4_2_profit_curve")

# %% =============================================================================
# STEP 4.3 - Threshold on the histogram
# ===============================================================================
header("STEP 4.3 - Histogram with threshold")


def threshold_histogram(ax, scores, threshold, title, label_color="decision", actual=None):
    """Bars left of the threshold = Bad Credit Risk (denied), right = Good Credit Risk.
    If label_color == 'actual', bars are stacked by the dataset's true class instead."""
    scores = np.asarray(scores)
    if label_color == "actual":
        ax.hist([scores[actual == 1], scores[actual == 0]], bins=BINS, stacked=True,
                color=[C_GOOD, C_BAD], edgecolor="white", linewidth=1.2,
                label=["Good credit risk (dataset)", "Bad credit risk (dataset)"])
    else:
        ax.hist([scores[scores >= threshold], scores[scores < threshold]], bins=BINS,
                stacked=True, color=[C_GOOD, C_BAD], edgecolor="white", linewidth=1.2,
                label=["Good Credit Risk (approved)", "Bad Credit Risk (denied)"])
    ax.axvline(threshold - 0.5 if ROUND_SCORES else threshold, color=C_LINE, linestyle="--", linewidth=1.5)
    ax.text(threshold + 1, ax.get_ylim()[1] * 0.95, f"Threshold = {threshold}", fontsize=9, va="top")
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("Creditworthiness")
    ax.set_ylabel("Members")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


fig, ax = plt.subplots(figsize=(7.5, 4))
threshold_histogram(ax, train["score"], BEST_T, "Creditworthiness with profit-maximising threshold")
save_fig(fig, "step4_3_histogram_threshold")

# Extra (optional) view: same histogram stacked by the dataset's actual class
fig, ax = plt.subplots(figsize=(7.5, 4))
threshold_histogram(ax, train["score"], BEST_T, "Creditworthiness by actual dataset class",
                    label_color="actual", actual=train["approved_actual"].to_numpy())
save_fig(fig, "step4_3_histogram_threshold_by_actual_class")

# %% =============================================================================
# STEP 4.4 - Favorable vs unfavorable outcomes per group
# ===============================================================================
header("STEP 4.4 - Favorable vs. unfavorable outcomes (single threshold)")
approve_single = (train["score"] >= BEST_T).to_numpy()
save_table(favorable_table(train, approve_single), "step4_4_outcomes_single_threshold")

# %% =============================================================================
# STEP 5.1 - 5.3 - Fairness metrics (training set, single threshold)
# ===============================================================================
header("STEP 5 - Fairness metrics (training set, single threshold)")
metrics_before, rates_before = compute_metrics(train, approve_single)
report(f"Approval rate privileged   ({PRIV_LABEL}): {rates_before[1]['fav']:.3f}")
report(f"Approval rate unprivileged ({UNPRIV_LABEL}): {rates_before[0]['fav']:.3f}")

# Context only: the same metrics computed on the dataset's own labels (no model)
metrics_dataset, _ = compute_metrics(train, train["approved_actual"].to_numpy())


def metrics_table(values, extra_cols=None):
    rows = []
    for k in FAIRNESS_METRICS:
        name, _, ideal, (lo, hi), formula = METRICS[k]
        row = {"Fairness metric": name, "Definition": formula,
               "Computed value (unpriv vs priv)": round(values[k], 4),
               "Ideal value": ideal, "Acceptable range": f"[{lo}, {hi}]",
               "Bias?": bias_verdict(k, values[k])}
        if extra_cols:
            row.update({c: round(v[k], 4) for c, v in extra_cols.items()})
        rows.append(row)
    return pd.DataFrame(rows)


report("\nStep 5.3 table:")
save_table(metrics_table(metrics_before,
                         {"(context) value on dataset labels": metrics_dataset}),
           "step5_3_fairness_metrics", index=False)


def plot_metrics(values_list, labels, name, title):
    """AIF360-style bar chart: one panel per metric, grey band = fair range."""
    fig, axes = plt.subplots(1, len(FAIRNESS_METRICS), figsize=(4.2 * len(FAIRNESS_METRICS), 3.6))
    colors = [C_BAD, C_GOOD, "#1baf7a"]
    for ax, k in zip(np.atleast_1d(axes), FAIRNESS_METRICS):
        mname, _, ideal, (lo, hi), _ = METRICS[k]
        vals = [v[k] for v in values_list]
        span = max(abs(hi - ideal), *[abs(v - ideal) for v in vals]) * 1.6
        ax.axhspan(lo, hi, color=C_FAIR, alpha=0.6, zorder=0, label="Fair range")
        ax.axhline(ideal, color=C_LINE, linewidth=1)
        x = np.arange(len(vals))
        bars = ax.bar(x, [v - ideal for v in vals], bottom=ideal, width=0.45,
                      color=colors[:len(vals)], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= ideal else "top", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.9, len(vals) - 0.1)
        ax.set_ylim(ideal - span, ideal + span)
        ax.text(1.02, (lo + hi) / 2 if k != "disparate_impact" else 1.0, "Fair",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.text(1.02, ideal - span * 0.85, "Bias", color="#b03a2e",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.set_title(mname, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{title}\nProtected attribute: {PROTECTED_ATTRIBUTE}  |  "
                 f"Privileged: {PRIV_LABEL}  |  Unprivileged: {UNPRIV_LABEL}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, name)


# Step 5.2
plot_metrics([metrics_before], [f"Threshold = {BEST_T}"], "step5_2_fairness_metrics",
             "Fairness metrics - training set, single threshold")

report("\nStep 5.4 helper (facts for your discussion paragraph):")
for k in FAIRNESS_METRICS:
    report(f"  {METRICS[k][0]} = {metrics_before[k]:.3f} -> {bias_verdict(k, metrics_before[k])}"
           f" (dataset labels alone: {metrics_dataset[k]:.3f})")

# %% =============================================================================
# STEP 6.1 - Initial (separate) thresholds
# ===============================================================================
header("STEP 6.1 - Initial thresholds per group")
priv_m = (train["privileged"] == 1).to_numpy()
unpriv_m = ~priv_m
scores = train["score"].to_numpy()


def approve_two(data, t_priv, t_unpriv):
    s = data["score"].to_numpy()
    p = (data["privileged"] == 1).to_numpy()
    return np.where(p, s >= t_priv, s >= t_unpriv)


T0_PRIV, T0_UNPRIV = INITIAL_THRESHOLDS if INITIAL_THRESHOLDS else (BEST_T, BEST_T)
appr0 = approve_two(train, T0_PRIV, T0_UNPRIV)
m0, _ = compute_metrics(train, appr0)
report(f"Starting thresholds: privileged = {T0_PRIV}, unprivileged = {T0_UNPRIV}")
report(f"  profit = {total_profit(train, appr0)}, "
       + ", ".join(f"{METRICS[k][0]} = {m0[k]:.3f}" for k in FAIRNESS_METRICS))

# %% =============================================================================
# STEP 6.2 - Search all threshold pairs: minimise bias while maximising profit
# ===============================================================================
header("STEP 6.2 - Optimise thresholds (training set)")
T = np.array(list(THRESHOLD_RANGE))
y = train["approved_actual"].to_numpy().astype(bool)


def per_group_curves(mask):
    s, yy = scores[mask], y[mask]
    A = s[None, :] >= T[:, None]                    # (n_thresholds, n_members)
    TP = (A & yy).sum(1); FP = (A & ~yy).sum(1)
    FN = (~A & yy).sum(1); TN = (~A & ~yy).sum(1)
    return dict(profit=TP * PROFIT_TP + FN * PROFIT_FN + FP * PROFIT_FP + TN * PROFIT_TN,
                fav=A.mean(1), tpr=TP / max(yy.sum(), 1), fpr=FP / max((~yy).sum(), 1),
                approve_all=A.all(1))


gp, gu = per_group_curves(priv_m), per_group_curves(unpriv_m)
# Grid: rows = privileged threshold, cols = unprivileged threshold
PROFIT_GRID = gp["profit"][:, None] + gu["profit"][None, :]
rates_grid = {1: {k: gp[k][:, None] for k in ("fav", "tpr", "fpr")},
              0: {k: gu[k][None, :] for k in ("fav", "tpr", "fpr")}}
with np.errstate(divide="ignore", invalid="ignore"):
    METRIC_GRID = np.broadcast_to(METRICS[MITIGATION_METRIC][1](rates_grid), PROFIT_GRID.shape).astype(float)
IDEAL = METRICS[MITIGATION_METRIC][2]
DEV_GRID = np.abs(METRIC_GRID - IDEAL)
ALLOWED = np.isfinite(DEV_GRID)
if FORBID_APPROVE_ALL:
    ALLOWED &= ~gp["approve_all"][:, None] & ~gu["approve_all"][None, :]

if OPTIMIZATION_MODE == "constrained":
    ok = ALLOWED & (DEV_GRID <= FAIRNESS_TOLERANCE)
    if ok.any():
        # highest profit; ties broken by smaller bias, then by closeness to the Step-4 threshold
        cand = np.argwhere(ok)
        key = [(-PROFIT_GRID[i, j], DEV_GRID[i, j], abs(T[i] - BEST_T) + abs(T[j] - BEST_T)) for i, j in cand]
        i, j = cand[min(range(len(cand)), key=lambda n: key[n])]
    else:
        report(f"  No threshold pair is within tolerance {FAIRNESS_TOLERANCE}; using least-biased pair.")
        d = np.where(ALLOWED, DEV_GRID, np.inf)
        i, j = np.unravel_index(np.argmin(d - 1e-9 * PROFIT_GRID), d.shape)
elif OPTIMIZATION_MODE == "weighted":
    obj = PROFIT_GRID / np.abs(PROFIT_GRID[ALLOWED]).max() - BIAS_WEIGHT * DEV_GRID
    obj = np.where(ALLOWED, obj, -np.inf)
    i, j = np.unravel_index(np.argmax(obj), obj.shape)
else:
    raise ValueError("OPTIMIZATION_MODE must be 'constrained' or 'weighted'")

T_PRIV, T_UNPRIV = int(T[i]), int(T[j])
approve_mit = approve_two(train, T_PRIV, T_UNPRIV)
profit_mit = total_profit(train, approve_mit)
metrics_after, rates_after = compute_metrics(train, approve_mit)
report(f"Optimisation: mode = {OPTIMIZATION_MODE}, metric = {METRICS[MITIGATION_METRIC][0]}"
       + (f", tolerance = {FAIRNESS_TOLERANCE}" if OPTIMIZATION_MODE == "constrained" else f", weight = {BIAS_WEIGHT}"))
report(f"Searched {ALLOWED.sum()} allowed threshold pairs out of {ALLOWED.size}.")
for name, t, g in (("privileged", T_PRIV, gp), ("unprivileged", T_UNPRIV, gu)):
    k = T_PRIV if name == "privileged" else T_UNPRIV
    if g["fav"][k - T[0]] >= 0.99:
        report(f"  WARNING: the {name} threshold approves >= 99% of that group - close to "
               "'approve all'. Consider changing the formula (see assignment note in Step 6.2).")

# Fairness-profit trade-off chart: for each level of bias, best achievable profit
dev_cap = max(0.3, 2.5 * abs(metrics_before[MITIGATION_METRIC] - IDEAL))   # x-axis range shown
dev_levels = np.linspace(0, dev_cap, 121)
frontier = np.array([np.max(np.where(ALLOWED & (DEV_GRID <= d), PROFIT_GRID, -np.inf)) for d in dev_levels])
frontier[~np.isfinite(frontier)] = np.nan
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(dev_levels, frontier, color=C_GOOD, linewidth=2, drawstyle="steps-post")
ax.scatter([abs(metrics_before[MITIGATION_METRIC] - IDEAL)], [best_profit], color=C_BAD, s=60,
           zorder=3, label=f"Single threshold {BEST_T}")
ax.scatter([abs(metrics_after[MITIGATION_METRIC] - IDEAL)], [profit_mit], color=C_LINE, s=60,
           marker="D", zorder=3, label=f"Chosen pair ({T_PRIV}, {T_UNPRIV})")
ax.set_xlabel(f"|{METRICS[MITIGATION_METRIC][0]} - {IDEAL}|  (bias allowed)")
ax.set_ylabel("Best achievable profit")
ax.set_xlim(0, dev_cap)
ax.set_title("Fairness / profit trade-off over all threshold pairs (training set)")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "step6_2_tradeoff_frontier")

# %% =============================================================================
# STEP 6.3 - Histograms per group with their thresholds
# ===============================================================================
header("STEP 6.3 - Histograms per group")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
threshold_histogram(axes[0], scores[priv_m], T_PRIV, f"Privileged: {PRIV_LABEL}")
threshold_histogram(axes[1], scores[unpriv_m], T_UNPRIV, f"Unprivileged: {UNPRIV_LABEL}")
fig.tight_layout()
save_fig(fig, "step6_3_histograms_by_group")

# Extra: same, stacked by actual dataset class
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
threshold_histogram(axes[0], scores[priv_m], T_PRIV, f"Privileged: {PRIV_LABEL} (actual class)",
                    label_color="actual", actual=y[priv_m].astype(int))
threshold_histogram(axes[1], scores[unpriv_m], T_UNPRIV, f"Unprivileged: {UNPRIV_LABEL} (actual class)",
                    label_color="actual", actual=y[unpriv_m].astype(int))
fig.tight_layout()
save_fig(fig, "step6_3_histograms_by_group_actual_class")

# %% =============================================================================
# STEP 6.4 / 6.5 - Mitigation results
# ===============================================================================
header("STEP 6.4 - Bias mitigation results (training set)")
report(f"1. Threshold privileged   ({PRIV_LABEL}): {T_PRIV}")
report(f"   Threshold unprivileged ({UNPRIV_LABEL}): {T_UNPRIV}")
report(f"2. Profit with these thresholds: {profit_mit}   "
       f"(single threshold {BEST_T}: {best_profit}; change {profit_mit - best_profit:+d})")

header("STEP 6.5 - Favorable vs. unfavorable outcomes (new thresholds)")
save_table(favorable_table(train, approve_mit), "step6_5_outcomes_new_thresholds")

report("\nFairness metrics before vs. after mitigation:")
save_table(metrics_table(metrics_after, {"Before (single threshold)": metrics_before}),
           "step6_5_fairness_metrics_after", index=False)
plot_metrics([metrics_before, metrics_after],
             [f"Before\n(X={BEST_T})", f"After\n({T_PRIV} / {T_UNPRIV})"],
             "step6_5_fairness_before_after", "Fairness metrics before vs. after mitigation (training set)")

# %% =============================================================================
# STEP 7 - Facts to support the post-mitigation discussion
# ===============================================================================
header("STEP 7 - Facts for the post-mitigation discussion")
for g, label in ((1, PRIV_LABEL), (0, UNPRIV_LABEL)):
    report(f"{label}: approval rate {rates_before[g]['fav']:.3f} -> {rates_after[g]['fav']:.3f}; "
           f"TPR {rates_before[g]['tpr']:.3f} -> {rates_after[g]['tpr']:.3f}; "
           f"FPR {rates_before[g]['fpr']:.3f} -> {rates_after[g]['fpr']:.3f}")
for g, label, m in ((1, PRIV_LABEL, priv_m), (0, UNPRIV_LABEL, unpriv_m)):
    pb = profit_from_counts(confusion(y[m], approve_single[m]))
    pa = profit_from_counts(confusion(y[m], approve_mit[m]))
    report(f"Profit from {label}: {pb} -> {pa}")
for k in FAIRNESS_METRICS:
    report(f"{METRICS[k][0]}: {metrics_before[k]:.3f} -> {metrics_after[k]:.3f} "
           f"({bias_verdict(k, metrics_after[k])})")

if EVALUATE_ON_TEST:
    report("\n(Extra) Same thresholds applied to the held-out TEST set:")
    a1 = (test["score"] >= BEST_T).to_numpy()
    a2 = approve_two(test, T_PRIV, T_UNPRIV)
    mt1, _ = compute_metrics(test, a1)
    mt2, _ = compute_metrics(test, a2)
    report(f"  Single threshold {BEST_T}: profit {total_profit(test, a1)}, "
           + ", ".join(f"{METRICS[k][0]} {mt1[k]:.3f}" for k in FAIRNESS_METRICS))
    report(f"  Thresholds ({T_PRIV}, {T_UNPRIV}): profit {total_profit(test, a2)}, "
           + ", ".join(f"{METRICS[k][0]} {mt2[k]:.3f}" for k in FAIRNESS_METRICS))

(OUTPUT_DIR / "results_summary.txt").write_text("\n".join(SUMMARY_LINES), encoding="utf-8")
print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
