import json
import pandas as pd
import numpy as np
import itertools


# ------------------------------------------------------
# Load JSON files (4 verifiers, 100 items)
# ------------------------------------------------------
paths = ["DeepSeekEval.json","Human1Eval.json","Human2Eval.json","Human3Eval.json"]
dfs = []
for p in paths:
    with open(p) as f:
        data = json.load(f)["Cases"]
    dfs.append(pd.DataFrame(data).set_index("number"))


# ------------------------------------------------------
# ORDINAL MAPPING FOR EACH METRIC
# ------------------------------------------------------
ordinal_maps = {
    "Correctness": {"Incorrect": 0, "Correct": 1},
    "Consistency with Report": {"Inconsistent": 0, "Partially Consistent": 1, "Fully Consistent": 2},
    "Answer Completeness": {"Incomplete": 0, "Partially Complete": 1, "Complete Answer": 2},
    "Clinical Relevance": {"Incorrect and Misleading": 0, "Non-essential but Correct": 1, "Essential": 2}
}


# ------------------------------------------------------
# QUADRATIC WEIGHT MATRIX
# ------------------------------------------------------
def quadratic_weights(C):
    W = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            W[i, j] = 1 - ((i - j)**2 / (C - 1)**2)
    return W


# ------------------------------------------------------
# Weighted Fleiss' Kappa
# ------------------------------------------------------
def fleiss_kappa_weighted(ratings, weights):
    ratings = np.asarray(ratings)
    n_items, n_raters = ratings.shape
    C = weights.shape[0]

    total_weighted = 0.0
    total_pairs = 0

    for i in range(n_items):
        item_ratings = ratings[i]
        for a, b in itertools.combinations(range(n_raters), 2):
            r1, r2 = item_ratings[a], item_ratings[b]
            total_weighted += weights[r1, r2]
            total_pairs += 1

    Po = total_weighted / total_pairs

    # Expected agreement
    p = np.bincount(ratings.flatten(), minlength=C) / ratings.size
    Pe = 0.0
    for i in range(C):
        for j in range(C):
            Pe += p[i] * p[j] * weights[i, j]

    if Pe == 1:
        return 1.0

    kappa = (Po - Pe) / (1 - Pe)
    return max(min(kappa, 1), -1)


# ------------------------------------------------------
# Gwet’s AC1 (nominal)
# ------------------------------------------------------
def gwet_ac1(ratings):
    ratings = np.asarray(ratings)
    n_items, n_raters = ratings.shape
    C = np.max(ratings) + 1

    total_equal = 0
    total_pairs = 0

    for i in range(n_items):
        item = ratings[i]
        for a, b in itertools.combinations(range(n_raters), 2):
            total_pairs += 1
            if item[a] == item[b]:
                total_equal += 1

    Po = total_equal / total_pairs

    p = np.bincount(ratings.flatten(), minlength=C) / ratings.size
    Pe = np.sum(p * (1 - p))

    return (Po - Pe) / (1 - Pe)


# ------------------------------------------------------
# Gwet’s AC2 (ordinal, weighted)
# ------------------------------------------------------
def gwet_ac2(ratings, weights):
    ratings = np.asarray(ratings)
    n_items, n_raters = ratings.shape
    C = weights.shape[0]

    # -----------------------------
    # Observed weighted agreement Po
    # -----------------------------
    total_weighted = 0.0
    total_pairs = 0

    for i in range(n_items):
        item = ratings[i]
        for a, b in itertools.combinations(range(n_raters), 2):
            total_weighted += weights[item[a], item[b]]
            total_pairs += 1

    Po = total_weighted / total_pairs

    # -----------------------------
    # Expected agreement Pe (AC2 definition)
    # -----------------------------
    p = np.bincount(ratings.flatten(), minlength=C) / ratings.size

    # compute w̄_j = Σ_i p_i * w_ij
    wbar = np.zeros(C)
    for j in range(C):
        wbar[j] = sum(p[i] * weights[i, j] for i in range(C))

    # q_j = 1 - p_j
    q = 1 - p

    # AC2 expected agreement
    Pe = sum(q[j] * wbar[j] for j in range(C)) / (C - 1)


    # -----------------------------
    # AC2 coefficient
    # -----------------------------
    return (Po - Pe) / (1 - Pe)


# ------------------------------------------------------
# MAIN LOOP — compute only Fleiss K, AC1, AC2
# ------------------------------------------------------
metrics = ["Correctness","Consistency with Report","Answer Completeness","Clinical Relevance"]
results = {}

for metric in metrics:
    # print("\n====", metric, "====")

    mapping = ordinal_maps[metric]

    df_ord = []
    for i in range(len(paths)):
        df_ord.append(dfs[i][metric].map(mapping).values)

    ratings = np.vstack(df_ord).T
    C = len(mapping)
    W = quadratic_weights(C)

    results[metric] = {
        "Quadratic Fleiss K": fleiss_kappa_weighted(ratings, W),
        "Gwet AC1": gwet_ac1(ratings),
        "Gwet AC2": gwet_ac2(ratings, W)
    }


# Print results
for m, vals in results.items():
    print("\n====", m, "====")
    for k, v in vals.items():
        print(f"{k}: {v:.4f}")
