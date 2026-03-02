
# BAYESIAN APPROACH 

from libraries import * 
from typing import Iterable
from itertools import product
def _to_list(cols):
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    if isinstance(cols, Iterable):
        return list(cols)
    raise ValueError("cols must be string or iterable of strings")

def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)

def build_probabilities_multi_bayes(df, x_col="X", y_col="Y", w_cols=None, z_cols=None,
                                   alpha_y=1.0, alpha_w=1.0, alpha_z=1.0):
    w_cols = _to_list(w_cols)
    z_cols = _to_list(z_cols)

    x_series = df[x_col]
    z_series = [df[c] for c in z_cols]
    w_series = [df[c] for c in w_cols]

    y_levels = pd.Index(df[y_col].unique()).sort_values()
    w_levels = {c: pd.Index(df[c].unique()).sort_values() for c in w_cols}
    z_levels = {c: pd.Index(df[c].unique()).sort_values() for c in z_cols}

    if z_cols:
        z_counts = df[z_cols].value_counts(dropna=False)
        P_z = (z_counts / z_counts.sum()).to_dict()
    else:
        P_z = {(): 1.0}

    # P(Y | X,Z,W) Bayesian 
    idx_cols_y_xzw = [x_series] + z_series + w_series if (z_cols or w_cols) else [x_series]

    # counts, not normalized
    ctab_y_xzw = pd.crosstab(index=idx_cols_y_xzw, columns=df[y_col], normalize=False)
    ctab_y_xzw = ctab_y_xzw.reindex(columns=y_levels, fill_value=0)

    def P_y_given_x_z_w(y, x, z_tuple, w_tuple):
        z_tuple = _to_tuple(z_tuple) if z_cols else ()
        w_tuple = _to_tuple(w_tuple) if w_cols else ()
        key = (x,) + z_tuple + w_tuple

        # row counts-if unobserved, treat as all zeros 
        if key in ctab_y_xzw.index:
            row = ctab_y_xzw.loc[key]
        else:
            row = pd.Series(0, index=ctab_y_xzw.columns)

        # posterior predictive with symmetric Dirichlet(alpha_y)
        N = float(row.sum())
        K = float(len(y_levels))
        num = float(row.get(y, 0)) + alpha_y
        den = N + alpha_y * K
        return num / den

    # P(Y | X,Z) Bayesian 
    idx_cols_y_xz = [x_series] + z_series if z_cols else [x_series]
    ctab_y_xz = pd.crosstab(index=idx_cols_y_xz, columns=df[y_col], normalize=False)
    ctab_y_xz = ctab_y_xz.reindex(columns=y_levels, fill_value=0)

    def P_y_given_x_z(y, x, z_tuple):
        if z_cols:
            z_tuple = _to_tuple(z_tuple)
            key = (x,) + z_tuple
        else:
            key = (x,)

        if key in ctab_y_xz.index:
            row = ctab_y_xz.loc[key]
        else:
            row = pd.Series(0, index=ctab_y_xz.columns)

        N = float(row.sum())
        K = float(len(y_levels))
        return (float(row.get(y, 0)) + alpha_y) / (N + alpha_y * K)


    if w_cols:
        idx_cols_w_xz = [x_series] + z_series
        ctab_w_xz = pd.crosstab(index=idx_cols_w_xz, columns=[df[c] for c in w_cols], normalize=False)
        # Compute joint levels list of tuples from observed combos:
        if len(w_cols) == 1:
            w_joint_levels = w_levels[w_cols[0]]
        else:
            w_joint_levels = pd.MultiIndex.from_frame(df[w_cols].drop_duplicates()).sort_values()

        ctab_w_xz = ctab_w_xz.reindex(columns=w_joint_levels, fill_value=0)

        def P_w_given_x_z(x, z_tuple, w_tuple):
            z_tuple = _to_tuple(z_tuple) if z_cols else ()
            w_tuple = _to_tuple(w_tuple)
            key_idx = (x,) + z_tuple

            key_col = w_tuple[0] if len(w_cols) == 1 else tuple(w_tuple)

            if key_idx in ctab_w_xz.index:
                row = ctab_w_xz.loc[key_idx]
            else:
                row = pd.Series(0, index=ctab_w_xz.columns)

            N = float(row.sum())
            K = float(len(row.index))
            return (float(row.get(key_col, 0)) + alpha_w) / (N + alpha_w * K)
    else:
        def P_w_given_x_z(x, z_tuple, w_tuple):
            return 1.0

    # P(Z | X) Bayesian
    if z_cols:
        ctab_z_x = pd.crosstab(index=x_series, columns=[df[c] for c in z_cols], normalize=False)

        if len(z_cols) == 1:
            z_joint_levels = z_levels[z_cols[0]]
        else:
            z_joint_levels = pd.MultiIndex.from_frame(df[z_cols].drop_duplicates()).sort_values()

        ctab_z_x = ctab_z_x.reindex(columns=z_joint_levels, fill_value=0)

        def P_z_given_x(z_tuple, x):
            z_tuple = _to_tuple(z_tuple)
            key_col = z_tuple[0] if len(z_cols) == 1 else tuple(z_tuple)

            if x in ctab_z_x.index:
                row = ctab_z_x.loc[x]
            else:
                row = pd.Series(0, index=ctab_z_x.columns)

            N = float(row.sum())
            K = float(len(row.index))
            return (float(row.get(key_col, 0)) + alpha_z) / (N + alpha_z * K)
    else:
        def P_z_given_x(z_tuple, x):
            return 1.0

    return (
        P_y_given_x_z,
        P_y_given_x_z_w,
        P_w_given_x_z,
        P_z_given_x,
        P_z,
        w_cols,
        z_cols,
    )

# CHANGE TO: build_probabilities_multi_bayes(..., alpha_y=1.0, alpha_w=1.0, alpha_z=1.0)







def make_ie_subset_fn_bayes(
    df,
    x_col,
    y_col,
    w_cols,
    z_cols=None,
    topo_order=None,
    y_val=1,
    alpha_w=1.0,          # Dirichlet smoothing for mediators
    alpha_y=1.0,          # Beta prior (success) for Y==y_val
    beta_y=1.0,           # Beta prior (failure) for Y==y_val
    alpha_z=None,         # if None -> empirical P(Z); else Dirichlet smoothing on observed Z tuples
    full_w_support=True  # if True -> Cartesian product of mediator levels (too big)
):

    topo   = list(topo_order) if topo_order is not None else list(w_cols)
    z_cols = _to_list(z_cols)
    if z_cols:
        Z_support = list(map(tuple, df[z_cols].drop_duplicates().itertuples(index=False, name=None)))
    else:
        Z_support = [()]
    w_levels = {w: list(pd.Index(df[w].drop_duplicates()).sort_values()) for w in topo}
    if not topo:
        W_support = [()]
    else:
        if full_w_support:
            # Cartesian product of per-mediator levels
            W_support = list(product(*[w_levels[w] for w in topo]))
        else:
            # Observed joint only
            W_support = list(map(tuple, df[topo].drop_duplicates().itertuples(index=False, name=None)))

    # P(Z)
    if not z_cols:
        P_z = {(): 1.0}
    else:
        z_counts = df[z_cols].value_counts(dropna=False)
        if alpha_z is None:
            # empirical
            P_z = (z_counts / z_counts.sum()).to_dict()
        else:
            # Dirichlet smoothing over observed Z-tuples
            z_tuples = list(z_counts.index)
            N = float(z_counts.sum())
            K = float(len(z_tuples))
            P_z = {zt: (float(z_counts.loc[zt]) + alpha_z) / (N + alpha_z * K) for zt in z_tuples}


    group_cols = [x_col] + (z_cols if z_cols else []) + topo
    df2 = df.copy()
    df2["__ybin__"] = (df2[y_col] == y_val).astype(int)

    g = df2.groupby(group_cols, dropna=False)["__ybin__"]
    sum_y = g.sum()
    n_y = g.count()

    def Ey_postpred(row_key_y):
        # Posterior predictive mean 
        s = float(sum_y.get(row_key_y, 0.0))
        n = float(n_y.get(row_key_y, 0.0))
        return (s + float(alpha_y)) / (n + float(alpha_y) + float(beta_y))

    # Mediator models: P(W_j | X,Z, previous W's) with Dirichlet smoothing
    med_counts = {}
    for j, wj in enumerate(topo):
        parent_cols = [x_col] + (z_cols if z_cols else []) + topo[:j]
        # counts table: rows=parents config, cols=values of wj
        ct = pd.crosstab(
            index=[df[c] for c in parent_cols],
            columns=df[wj],
            normalize=False,
            dropna=False,
        )
        levels = pd.Index(w_levels[wj])
        ct = ct.reindex(columns=levels, fill_value=0)
        med_counts[wj] = (ct, levels)

    def P_wj_postpred(wj, row_key, wj_value):
        
        ct, levels = med_counts[wj]
        if row_key in ct.index:
            row = ct.loc[row_key]
        else:
            row = pd.Series(0, index=ct.columns)

        K = float(len(levels))
        N = float(row.sum())
        c = float(row.get(wj_value, 0.0))
        return (c + float(alpha_w)) / (N + float(alpha_w) * K)

    # Hybrid assignment policy 
    def _Ey_policy(x_for_Y, x_for_each_w):
        total = 0.0
        for z in Z_support:
            pz = P_z[z if z_cols else ()]
            inner = 0.0

            for w_tuple in W_support:
                # product over mediators under the hybrid policy
                p_w = 1.0
                prev_vals = ()

                for j, wj in enumerate(topo):
                    x_seen = x_for_each_w[wj]
                    # row_key for mediator j: (x, z..., previous w's)
                    row_key = (x_seen,) + (z if z_cols else ()) + prev_vals
                    p_wj = P_wj_postpred(wj, row_key, w_tuple[j])
                    p_w *= p_wj
                    if p_w == 0.0:
                        break
                    prev_vals = prev_vals + (w_tuple[j],)

                if p_w == 0.0:
                    continue

                row_key_y = (x_for_Y,) + (z if z_cols else ()) + w_tuple
                Ey = Ey_postpred(row_key_y)

                inner += p_w * Ey

            total += pz * inner

        return float(total)

    # IE_REVERSE + decomposition 
    def call_fn(x0, x1, S=None):
        Ey_x1    = _Ey_policy(x_for_Y=x1, x_for_each_w={w: x1 for w in topo})
        Ey_x1Wx0 = _Ey_policy(x_for_Y=x1, x_for_each_w={w: x0 for w in topo})
        IE       = Ey_x1Wx0 - Ey_x1

        if S is not None:
            S = set(S)
            Ey_subset = _Ey_policy(
                x_for_Y=x1,
                x_for_each_w={w: (x0 if w in S else x1) for w in topo},
            )
            return {"Ey_x1": Ey_x1, "Ey_x1_Wx0": Ey_x1Wx0, "Ey_subset": Ey_subset, "IE": IE}
        # If S is None: compute stepwise contributions in topo order
        contribs = []
        subset_E = {}
        prev_E = Ey_x1
        subset_E[()] = prev_E
        turned_to_x0 = []

        for w in topo:
            turned_to_x0.append(w)
            curr_policy = {ww: (x0 if ww in turned_to_x0 else x1) for ww in topo}
            curr_E = _Ey_policy(x_for_Y=x1, x_for_each_w=curr_policy)
            subset_E[tuple(turned_to_x0)] = curr_E
            contribs.append(curr_E - prev_E)
            prev_E = curr_E

        return {
            "Ey_x1": Ey_x1,
            "Ey_x1_Wx0": Ey_x1Wx0,
            "subset_expectations": subset_E,
            "variable_contributions": pd.Series(contribs, index=topo),
            "IE": IE,
        }

    return call_fn



