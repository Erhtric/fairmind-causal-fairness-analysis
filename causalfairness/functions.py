from general_effects import *
from libraries import *
from bayesian import *
from typing import Iterable
import itertools
import numpy as np


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


# FREQUENTIST APPROACH: Not used. 
def build_probabilities_multi(df, x_col="X", y_col="Y",
                              w_cols=None, z_cols=None):

    w_cols = _to_list(w_cols)
    z_cols = _to_list(z_cols)

    x_series = df[x_col]
    z_series = [df[c] for c in z_cols]
    w_series = [df[c] for c in w_cols]

    #P(Z = z_vec)
    if z_cols:
        P_z = df[z_cols].value_counts(normalize=True).to_dict()
    else:
        P_z = {(): 1.0}

    #P(Y | X, Z, W)
    if z_cols or w_cols:
        idx_cols_y_xzw = [x_series] + z_series + w_series
    else:
        idx_cols_y_xzw = [x_series]

    cpt_y_given_xzw = pd.crosstab(
        index=idx_cols_y_xzw,
        columns=df[y_col],
        normalize="index",
    )

    def P_y_given_x_z_w(y, x, z_tuple, w_tuple):
        z_tuple = _to_tuple(z_tuple) if z_cols else ()
        w_tuple = _to_tuple(w_tuple) if w_cols else ()
        key = (x,) + z_tuple + w_tuple
        try:
            return float(cpt_y_given_xzw.loc[key, y])
        except KeyError:
        #(x,z,w) never observed  -> probability treated as 0
            return 0.0

    #P(Y | X, Z)
    if z_cols:
        idx_cols_y_xz = [x_series] + z_series
    else:
        idx_cols_y_xz = [x_series]

    cpt_y_given_xz = pd.crosstab(
        index=idx_cols_y_xz,
        columns=df[y_col],
        normalize="index",
    )

    def P_y_given_x_z(y, x, z_tuple):
        if z_cols:
            z_tuple = _to_tuple(z_tuple)
            key = (x,) + z_tuple
        else:
            key = (x,)
        try:
            return float(cpt_y_given_xz.loc[key, y])
        except KeyError:
            return 0.0

    #P(W | X, Z)
    if w_cols:
        idx_cols_w_xz = [x_series] + z_series
        cpt_w_given_xz = pd.crosstab(
            index=idx_cols_w_xz,
            columns=[df[c] for c in w_cols], 
            normalize="index",
        )

        def P_w_given_x_z(x, z_tuple, w_tuple):
            z_tuple = _to_tuple(z_tuple) if z_cols else ()
            w_tuple = _to_tuple(w_tuple)
            key_idx = (x,) + z_tuple
            key_col = w_tuple[0] if len(w_cols) == 1 else w_tuple
            try:
                return float(cpt_w_given_xz.loc[key_idx, key_col])

            except KeyError:
                return 0.0
    else:
        def P_w_given_x_z(x, z_tuple, w_tuple):
            return 1.0

    #P(Z | X)
    if z_cols:
        cpt_z_given_x = pd.crosstab(
            index=x_series,
            columns=[df[c] for c in z_cols],
            normalize="index",
        )

        def P_z_given_x(z_tuple, x):
            z_tuple = _to_tuple(z_tuple)
            try:
                #return float(cpt_z_given_x.loc[x, z_tuple])
                key_col = z_tuple[0] if len(z_cols) == 1 else z_tuple
                return float(cpt_z_given_x.loc[x, key_col])

            except KeyError:
                return 0.0
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





#Decompose indirect effect, NEW

def decompose_ie_by_mediator_counts(ie_subset_fn, mediators, x0, x1):
    contrib = {}
    prev = ie_subset_fn(x0=x0, x1=x1, S=[] )["Ey_x1"]  # baseline E[Y_{x1}]
    active = []
    for w in mediators:
        active.append(w)
        curr = ie_subset_fn(x0=x0, x1=x1, S=active)["Ey_subset"]
        contrib[w] = curr - prev
        prev = curr
    return contrib

# Frequentist approach: not used

def make_ie_subset_fn_counts(df, x_col, y_col, w_cols, z_cols=None, topo_order=None,y_val=1):
    topo   = list(topo_order) if topo_order is not None else list(w_cols)
    z_cols = _to_list(z_cols)

    Z_support = [()] if not z_cols else list(
        map(tuple, df[z_cols].drop_duplicates().itertuples(index=False, name=None))
    )
    W_support = list(
        map(tuple, df[topo].drop_duplicates().itertuples(index=False, name=None))
    )

    P_z = df[z_cols].value_counts(normalize=True).to_dict() if z_cols else {(): 1.0}

    idx_y = [df[x_col]] + ([df[c] for c in z_cols] if z_cols else []) + [df[c] for c in topo]
    group_cols = [x_col] + (z_cols if z_cols else []) + topo

    df2 = df.copy()
    df2["__ybin__"] = (df2[y_col] == y_val).astype(float)  
    Ey_given_xzw = df2.groupby(group_cols)["__ybin__"].mean()



    med_cpts = {}
    for j, wj in enumerate(topo):
        parents = [df[x_col]] + ([df[c] for c in z_cols] if z_cols else []) + [df[c] for c in topo[:j]]
        med_cpts[wj] = pd.crosstab(index=parents, columns=df[wj], normalize="index")

    #Hybrid assignment
    def _Ey_policy(x_for_Y, x_for_each_w):
        total = 0.0
        for z in ([()] if not z_cols else Z_support):
            pz = P_z[z if z_cols else ()]
            inner = 0.0
            for w_tuple in W_support:
                p_w = 1.0
                prev_vals = ()
                for j, wj in enumerate(topo):
                    x_seen = x_for_each_w[wj] #hybrid switch for mediator wj
                    row_key = (x_seen,) + (z if z_cols else ()) + prev_vals
                    try:
                        p_wj = float(med_cpts[wj].loc[row_key, w_tuple[j]])
                    except KeyError:
                        p_wj = 0.0
                    p_w *= p_wj
                    if p_w == 0.0:
                        break
                    prev_vals = prev_vals + (w_tuple[j],)
                if p_w == 0.0:
                    continue  
                row_key_y = (x_for_Y,) + (z if z_cols else ()) + w_tuple
                try:
                    Ey = float(Ey_given_xzw.loc[row_key_y])
                except KeyError:
                    Ey = 0.0
                inner += p_w * Ey
                                
            total += pz * inner
        return float(total)

    #IE_REVERSE
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

        contribs = []
        subset_E = {}
        prev_policy = {w: x1 for w in topo}
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


#If No ORDERS: intervals.
def ie_minmax_intervals(df, x_col, y_col, w_cols, z_cols, x0, x1, y_val):
    w_cols = list(w_cols)
    mins = {w: np.inf for w in w_cols}
    maxs = {w: -np.inf for w in w_cols}

    for order in itertools.permutations(w_cols):
        ie_subset_fn = make_ie_subset_fn_counts(
            df=df,
            x_col=x_col,
            y_col=y_col,
            w_cols=w_cols,
            z_cols=z_cols,
            topo_order=order,
            y_val=y_val,
           # alpha_y=1.0, alpha_w=1.0, alpha_z=1.0
        )
        out = ie_subset_fn(x0=x0, x1=x1, S=None)
        vc = out["variable_contributions"]
        for w in w_cols:
            c = float(vc.loc[w])
            mins[w] = min(mins[w], c)
            maxs[w] = max(maxs[w], c)
    return {w: (float(mins[w]), float(maxs[w])) for w in w_cols}

#Decompose spuroius effects
def decompose_se_by_confounder(se_subset_fn, z_cols):
    contrib = {}
    prev = se_subset_fn([])  
    active = []

    for z in z_cols:
        active.append(z)
        curr = se_subset_fn(active)   
        contrib[z] = curr - prev
        prev = curr

    return contrib



def make_se_subset_fn(df, x0, x1, y, w_cols=None,
                      x_col="X", y_col="Y"):

    def se_subset(active_z_cols):
        active_z_cols = list(active_z_cols)

        (
            P_y_given_x_z,
            P_y_given_x_z_w,
            P_w_given_x_z,
            P_z_given_x,
            P_z,
            _w_cols,
            _z_cols,
        ) = build_probabilities_multi(
            df,
            x_col=x_col,
            y_col=y_col,
            w_cols=w_cols,
            z_cols=active_z_cols,
            #alpha_y=1.0, alpha_w=1.0, alpha_z=1.0
            )
        

        se_x1 = spurious_effect(
            x=x1,
            y=y,
            P_y_given_x_z=P_y_given_x_z,
            P_z=P_z,
            P_z_given_x=P_z_given_x,
        )

        se_x0 = spurious_effect(
            x=x0,
            y=y,
            P_y_given_x_z=P_y_given_x_z,
            P_z=P_z,
            P_z_given_x=P_z_given_x,
        )

        return se_x1 - se_x0   # SAME SE definition as in compute_effects_multi

    return se_subset

#IF NO ORDER: intervals


def se_minmax_intervals(se_subset_fn, z_cols):
    z_cols = list(z_cols)
    mins = {z: np.inf for z in z_cols}
    maxs = {z: -np.inf for z in z_cols}

    for order in itertools.permutations(z_cols):
        prev = se_subset_fn([])  
        active = []
        for z in order:
            active.append(z)
            curr = se_subset_fn(active)
            contrib = curr - prev
            mins[z] = min(mins[z], contrib)
            maxs[z] = max(maxs[z], contrib)
            prev = curr

    return {z: (float(mins[z]), float(maxs[z])) for z in z_cols}





def compute_effects_multi(
    df,
    x0=0,
    x1=1,
    y=1,
    W_values=None,
    x_col="X",
    y_col="Y",
    w_cols=None,
    z_cols=None,
    w_order=None,
    z_order=None,
    do_decomposition=True,
):
    (
        P_y_given_x_z,
        P_y_given_x_z_w,
        P_w_given_x_z,
        P_z_given_x,
        P_z,
        w_cols_list,
        z_cols_list,
    ) = build_probabilities_multi(
        df,
        x_col=x_col,
        y_col=y_col,
        w_cols=w_cols,
        z_cols=z_cols,
        #alpha_y=1.0, alpha_w=1.0, alpha_z=1.0
    )

    if W_values is None:
        if w_cols_list:
            if len(w_cols_list) == 1:
                vals = df[w_cols_list[0]].drop_duplicates()
                W_values = [(v,) for v in vals]
            else:
                W_values = list(
                    map(tuple, df[w_cols_list].drop_duplicates().itertuples(index=False, name=None))
                )
        else:
            W_values = [()]

    te = total_effect(x0=x0, x1=x1, y=y, P_y_given_x_z=P_y_given_x_z, P_z=P_z)

    se_x1 = spurious_effect(x=x1, y=y, P_y_given_x_z=P_y_given_x_z, P_z=P_z, P_z_given_x=P_z_given_x)
    se_x0 = spurious_effect(x=x0, y=y, P_y_given_x_z=P_y_given_x_z, P_z=P_z, P_z_given_x=P_z_given_x)
    se = se_x1 - se_x0
    tv = te + se
    tv1 = tv_effect(
    x0=x0, x1=x1, y=y,
    P_y_given_x_z=P_y_given_x_z,   
    P_z=P_z,
    P_z_given_x=P_z_given_x
)
    de = dir_effect(
        x0=x0, x1=x1, y=y, W_values=W_values,
        P_y_given_x_z_w=P_y_given_x_z_w, P_z=P_z, P_w_given_x_z=P_w_given_x_z
    )

    ie = ind_effect(
        x1=x1, x0=x0, y=y, W_values=W_values,
        P_y_given_x_z_w=P_y_given_x_z_w, P_w_given_x_z=P_w_given_x_z, P_z=P_z
    )

    ie_f = ind_f_effect(
        x0=x0, x1=x1, y=y, W_values=W_values,
        P_y_given_x_z_w=P_y_given_x_z_w, P_w_given_x_z=P_w_given_x_z, P_z=P_z
    )

    te_linear = de + ie_f

    ie_decomp = {}
    se_decomp = {}
    ie_decomp_interval = {}
    se_decomp_interval = {}

    # chosen order if provided
    w_order_used = list(w_order) if w_order else list(w_cols_list)
    z_order_used = list(z_order) if z_order else list(z_cols_list)

    if w_cols_list and set(w_order_used) != set(w_cols_list):
        raise ValueError("w_order must contain exactly the same variables as w_cols")
    if z_cols_list and set(z_order_used) != set(z_cols_list):
        raise ValueError("z_order must contain exactly the same variables as z_cols")

    if do_decomposition:
        # IE decomposition 
        if len(w_cols_list) > 1:
            if w_order is not None:
                ie_subset_fn = make_ie_subset_fn_counts(
                    df=df, x_col=x_col, y_col=y_col,
                    w_cols=w_cols_list, z_cols=z_cols_list,
                    topo_order=w_order_used, y_val=y,#alpha_y=1.0, alpha_w=1.0, alpha_z=1.0
                    )
        
                ie_decomp = decompose_ie_by_mediator_counts(ie_subset_fn, w_order_used, x0, x1)
            else:
                ie_decomp_interval = ie_minmax_intervals(
                    df=df, x_col=x_col, y_col=y_col,
                    w_cols=w_cols_list, z_cols=z_cols_list,
                    x0=x0, x1=x1, y_val=y
                )

        # SE decomposition
        if len(z_cols_list) > 1:
            se_subset_fn = make_se_subset_fn(
                df=df, x0=x0, x1=x1, y=y,
                w_cols=w_cols_list,
                x_col=x_col, y_col=y_col,
            )
            if z_order is not None:
                se_decomp = decompose_se_by_confounder(se_subset_fn, z_order_used)
            else:
                se_decomp_interval = se_minmax_intervals(se_subset_fn, z_cols_list)

    return {
        "tv": tv,
        "tv1":tv1,
        "te": te,
        "te_linear": te_linear,
        "se": se,
        "se_x1": se_x1,
        "se_x0": se_x0,
        "de": de,
        "ie": ie,
        "ie_f": ie_f,
        "ie_decomp": ie_decomp,
        "se_decomp": se_decomp,
        "ie_decomp_interval": ie_decomp_interval,
        "se_decomp_interval": se_decomp_interval,
    }

#x-specific 

def compute_x_specific_effects(
    df,
    x0,
    x1,
    x_cond,
    y_val,
    x_col="X",
    y_col="Y",
    w_cols=None,
    z_cols=None,
):
    (
        P_y_given_x_z,
        P_y_given_x_z_w,
        P_w_given_x_z,
        P_z_given_x,
        P_z,
        w_cols_list,
        z_cols_list,
    ) = build_probabilities_multi(
        df,
        x_col=x_col,
        y_col=y_col,
        w_cols=w_cols,
        z_cols=z_cols,
      #  alpha_y=1.0, alpha_w=1.0, alpha_z=1.0
    )

    #Z
    z_cols_list = _to_list(z_cols_list)
    if z_cols_list:
        z_tuples = {
            _to_tuple(t)
            for t in df[z_cols_list].itertuples(index=False, name=None)
        }
        z_tuples = list(z_tuples)
    else:
        z_tuples = [()]  # no Z

    #W
    w_cols_list = _to_list(w_cols_list)
    if w_cols_list:
        w_tuples = {
            _to_tuple(t)
            for t in df[w_cols_list].itertuples(index=False, name=None)
        }
        w_tuples = list(w_tuples)
    else:
        w_tuples = [()]  # no W

    #x-TE
    te_x = 0.0
    for z in z_tuples:
        te_x += (
            P_y_given_x_z(y_val, x1, z)
            - P_y_given_x_z(y_val, x0, z)
        ) * P_z_given_x(z, x_cond)

    #x-DE_x
    de_x = 0.0
    for z in z_tuples:
        for w in w_tuples:
            de_x += (
                P_y_given_x_z_w(y_val, x1, z, w)
                - P_y_given_x_z_w(y_val, x0, z, w)
            ) * P_w_given_x_z(x0, z, w) * P_z_given_x(z, x_cond)

    #x-IE
    ie_x = 0.0
    for z in z_tuples:
        for w in w_tuples:
            ie_x += (
                P_y_given_x_z_w(y_val, x1, z, w)
                * (P_w_given_x_z(x0, z, w) - P_w_given_x_z(x1, z, w))
                * P_z_given_x(z, x_cond)
            )
            #sto calcolando il ie x1,x0 (reverse)
    #x-SE
    se_x = 0.0
    for z in z_tuples:
        se_x += P_y_given_x_z(y_val, x0, z) * (
            P_z_given_x(z, x0) - P_z_given_x(z, x_cond)
        )

    return te_x, ie_x, de_x, se_x

#z-specific effect

def compute_z_specific_effects(
    df,
    x0,
    x1,
    y_val,
    x_col="X",
    y_col="Y",
    w_cols=None,
    z_cols=None,
):
    (
        P_y_given_x_z,
        P_y_given_x_z_w,
        P_w_given_x_z,
        P_z_given_x,
        P_z,
        w_cols_list,
        z_cols_list,
    ) = build_probabilities_multi(
        df,
        x_col=x_col,
        y_col=y_col,
        w_cols=w_cols,
        z_cols=z_cols,
       # alpha_y=1.0, alpha_w=1.0, alpha_z=1.0
       )
    

    z_cols_list = _to_list(z_cols_list)
    if z_cols_list:
        z_tuples = {
            _to_tuple(t)
            for t in df[z_cols_list].itertuples(index=False, name=None)
        }
        z_tuples = list(z_tuples)
    else:
        z_tuples = [()]  # no Z

    w_cols_list = _to_list(w_cols_list)
    if w_cols_list:
        w_tuples = {
            _to_tuple(t)
            for t in df[w_cols_list].itertuples(index=False, name=None)
        }
        w_tuples = list(w_tuples)
    else:
        w_tuples = [()]  # no W

    effects_by_z = {}

    for z in z_tuples:
        #z-TE
        z_TE = P_y_given_x_z(y_val, x1, z) - P_y_given_x_z(y_val, x0, z)

        #z-DE
        z_DE = 0.0
        for w in w_tuples:
            z_DE += (
                P_y_given_x_z_w(y_val, x1, z, w)
                - P_y_given_x_z_w(y_val, x0, z, w)
            ) * P_w_given_x_z(x0, z, w)

        #z-IE
        z_IE = 0.0
        for w in w_tuples:
            z_IE += (
                P_y_given_x_z_w(y_val, x1, z, w)
                * (P_w_given_x_z(x0, z, w) - P_w_given_x_z(x1, z, w))
            )

        effects_by_z[z] = {"z_TE": z_TE, "z_DE": z_DE, "z_IE": z_IE}

    return effects_by_z



def compute_effects_continuous_y(
    df,
    x_col,
    y_col,
    w_cols,
    z_cols,
    x0_value,
    x1_values,
    y_thresholds,
    w_order=None,
    z_order=None,
    progress_cb=None,   
):
    out = []
    x_bin_col = "__Xbin__"
    y_bin_col = "__Ybin__"
    

    for i, thr in enumerate(y_thresholds):
        df_cs = df.copy()
        df_cs[y_bin_col] = (df_cs[y_col] <= thr).astype(int)

        # keep only x0 and x1_values
        keep = df_cs[x_col].eq(x0_value) | df_cs[x_col].isin(x1_values)
        df_cs = df_cs.loc[keep].copy()

        # binarize for x1_values
        df_cs[x_bin_col] = df_cs[x_col].isin(x1_values).astype(int)

        eff = compute_effects_multi(
            df=df_cs,
            x0=0, x1=1, y=1,
            x_col=x_bin_col,
            y_col=y_bin_col,
            w_cols=w_cols,
            z_cols=z_cols,
            w_order=w_order,
            z_order=z_order,
            do_decomposition=False,
        )


        if hasattr(eff, "to_dict"):
            eff = eff.to_dict()

        # TV(y) is identifiable directly from data for the binary indicator
        p1 = df_cs.loc[df_cs[x_bin_col] == 1, y_bin_col].mean() if (df_cs[x_bin_col] == 1).any() else np.nan
        p0 = df_cs.loc[df_cs[x_bin_col] == 0, y_bin_col].mean() if (df_cs[x_bin_col] == 0).any() else np.nan

        tv = p1 - p0
        out.append({
            "y_threshold": float(thr),
            "tv": float(tv) if tv == tv else None,  # NaN safe
            "te": eff.get("te"),
            "de": eff.get("de"),
            "ie": eff.get("ie"),
            "se": eff.get("se"),
            "ie_decomp": None,
            "se_decomp": None,
        })
        if progress_cb is not None:
             progress_cb(i + 1, len(y_thresholds), float(thr))

    return out


def compute_stepwise_effects_continuous_y(
    df,
    x_col,
    y_col,
    w_cols,
    z_cols,
    y_thresholds,
    x_order,
    w_order=None,
    z_order=None,
):
    if not x_order or len(x_order) < 2:
        return []

    results = []
    for thr in y_thresholds:
        steps = []
        for i in range(len(x_order) - 1):
            x_from = x_order[i]
            x_to = x_order[i + 1]

            df_step = df[df[x_col].isin([x_from, x_to])].copy()
            if df_step.empty:
                steps.append({"from": x_from, "to": x_to, "n_rows": 0,
                              "tv": None, "te": None, "de": None, "ie": None, "se": None})
                continue

            x_bin = "__Xbin_step__"
            y_bin = "__Ybin_step__"
            df_step[x_bin] = (df_step[x_col] == x_to).astype(int)
            df_step[y_bin] = (df_step[y_col] <= thr).astype(int)

            eff = compute_effects_multi(
                df=df_step,
                x0=0, x1=1, y=1,
                x_col=x_bin,
                y_col=y_bin,
                w_cols=w_cols,
                z_cols=z_cols,
                w_order=w_order,
                z_order=z_order,
                do_decomposition=False,
            )
            if hasattr(eff, "to_dict"):
                eff = eff.to_dict()

            p1 = df_step.loc[df_step[x_bin] == 1, y_bin].mean() if (df_step[x_bin] == 1).any() else np.nan
            p0 = df_step.loc[df_step[x_bin] == 0, y_bin].mean() if (df_step[x_bin] == 0).any() else np.nan
            tv = (p1 - p0) if (p1 == p1 and p0 == p0) else None

            steps.append({
                "from": x_from, "to": x_to, "n_rows": int(df_step.shape[0]),
                "tv": tv if tv is not None else eff.get("tv"),
                "te": eff.get("te"),
                "de": eff.get("de"),
                "ie": eff.get("ie"),
                "se": eff.get("se"),
            })

        def _sum(key):
            vals = [s.get(key) for s in steps if s.get(key) is not None]
            return float(np.sum(vals)) if vals else None

        results.append({
            "y_threshold": float(thr),
            "effects_by_step": steps,
            "cumulative": {"tv": _sum("tv"), "te": _sum("te"), "de": _sum("de"), "ie": _sum("ie"), "se": _sum("se")}
        })

    return results


def rounded_val(v, nd=5):
    if v is None:
        return "—"
    if isinstance(v, (tuple, list)) and len(v) == 2:
        a, b = v
        try:
            return f"({float(a):.{nd}f}, {float(b):.{nd}f})"
        except Exception:
            return f"({a}, {b})"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)