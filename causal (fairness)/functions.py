from general_effects import *
from libraries import *

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
            key_col = w_tuple
            try:
                return float(cpt_w_given_xz.loc[key_idx, key_col])

            except KeyError:
                return 0.0
    else:
        def P_w_given_x_z(x, z_tuple, w_tuple):
            return 1.0

    #P(Z | X) (joint Z-vector)
    if z_cols:
        cpt_z_given_x = pd.crosstab(
            index=x_series,
            columns=[df[c] for c in z_cols],
            normalize="index",
        )

        def P_z_given_x(z_tuple, x):
            z_tuple = _to_tuple(z_tuple)
            try:
                return float(cpt_z_given_x.loc[x, z_tuple])
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





#Decompose indirect effect

def decompose_nie_by_mediator(nie_subset_fn, w_cols):
    contrib = {}
    prev = 0.0
    active = []

    for w in w_cols:
        active.append(w)
        curr = nie_subset_fn(active)   
        contrib[w] = curr - prev       
        prev = curr

    return contrib


def decompose_se_by_confounder(se_subset_fn, z_cols):
    contrib = {}
    prev = 0.0
    active = []

    for z in z_cols:
        active.append(z)
        curr = se_subset_fn(active)   
        contrib[z] = curr - prev
        prev = curr

    return contrib


def make_nie_subset_fn(df, x0, x1, y,
                       all_w_cols, z_cols,
                       W_values=(0, 1),
                       x_col="X", y_col="Y"):
    def nie_subset(active_w_cols):
        active_w_cols = list(active_w_cols)

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
            w_cols=active_w_cols,
            z_cols=z_cols,
        )

        nie_val = nat_ind_effect(
            x0=x0,
            x1=x1,
            y=y,
            W_values=W_values,
            P_y_given_x_z_w=P_y_given_x_z_w,
            P_w_given_x_z=P_w_given_x_z,
            P_z=P_z,
        )
        return nie_val

    return nie_subset

#Decompose spuroius effects
def make_se_subset_fn(df, x0, x1, y,
                      all_z_cols, w_cols,
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


#Compute effects
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
    )

    # If W_values not given, take all observed joint W combinations
    if W_values is None and w_cols_list:
        W_values = list(map(tuple, df[w_cols_list].drop_duplicates().values))
    elif W_values is None:
        W_values = [()]

    #Total effect
    te = total_effect(
        x0=x0,
        x1=x1,
        y=y,
        P_y_given_x_z=P_y_given_x_z,
        P_z=P_z,
    )

    #Spurious effect
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

    se = se_x1 - se_x0
    tv = te + se

    #Natural direct effect 
    nde = nat_dir_effect(
        x0=x0,
        x1=x1,
        y=y,
        W_values=W_values,
        P_y_given_x_z_w=P_y_given_x_z_w,
        P_z=P_z,
        P_w_given_x_z=P_w_given_x_z,
    )

    #Natural indirect effect (reverse)
    nie = nat_ind_effect(
        x1=x1,
        x0=x0,
        y=y,
        W_values=W_values,
        P_y_given_x_z_w=P_y_given_x_z_w,
        P_w_given_x_z=P_w_given_x_z,
        P_z=P_z,
    )

    #Natural indirect effect (forward)
    nie_f = nat_ind_f_effect(
        x0=x0,
        x1=x1,
        y=y,
        W_values=W_values,
        P_y_given_x_z_w=P_y_given_x_z_w,
        P_w_given_x_z=P_w_given_x_z,
        P_z=P_z,
    )

    te_linear = nde + nie_f 

    nie_decomp = {}
    se_decomp = {}

    if do_decomposition:
        if len(w_cols_list) > 1:
            nie_subset_fn = make_nie_subset_fn(
                df,
                x0=x0,
                x1=x1,
                y=y,
                all_w_cols=w_cols_list,
                z_cols=z_cols_list,
                W_values=W_values,
                x_col=x_col,
                y_col=y_col,
            )
            nie_decomp = decompose_nie_by_mediator(nie_subset_fn, w_cols_list)

        if len(z_cols_list) > 1:
            se_subset_fn = make_se_subset_fn(
                df,
                x0=x0,
                x1=x1,   
                y=y,
                all_z_cols=z_cols_list,
                w_cols=w_cols_list,
                x_col=x_col,
                y_col=y_col,
            )
            se_decomp = decompose_se_by_confounder(se_subset_fn, z_cols_list)

    return {
        "tv": tv,
        "te": te,
        "te_linear": te_linear,
        "se": se,
        "se_x1": se_x1,
        "se_x0": se_x0,
        "nde": nde,
        "nie":nie,
        "nie_f": nie_f,
        "nie_decomp": nie_decomp,
        "se_decomp": se_decomp,
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

    #x-NDE_x: Ctf-DE_{x0,x1}(y | x_cond) -----
    nde_x = 0.0
    for z in z_tuples:
        for w in w_tuples:
            nde_x += (
                P_y_given_x_z_w(y_val, x1, z, w)
                - P_y_given_x_z_w(y_val, x0, z, w)
            ) * P_w_given_x_z(x0, z, w) * P_z_given_x(z, x_cond)

    #x-NIE
    nie_x = 0.0
    for z in z_tuples:
        for w in w_tuples:
            nie_x += (
                P_y_given_x_z_w(y_val, x1, z, w)
                * (P_w_given_x_z(x0, z, w) - P_w_given_x_z(x1, z, w))
                * P_z_given_x(z, x_cond)
            )
            #sto calcolando il nie x1,x0 (reverse)
    #x-SE
    se_x = 0.0
    for z in z_tuples:
        se_x += P_y_given_x_z(y_val, x0, z) * (
            P_z_given_x(z, x0) - P_z_given_x(z, x1)
        )

    return te_x, nie_x, nde_x, se_x

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
                P_y_given_x_z_w(y_val, x0, z, w)
                * (P_w_given_x_z(x1, z, w) - P_w_given_x_z(x0, z, w))
            )

        effects_by_z[z] = {"z_TE": z_TE, "z_DE": z_DE, "z_IE": z_IE}

    return effects_by_z
