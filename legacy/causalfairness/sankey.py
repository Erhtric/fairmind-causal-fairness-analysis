from libraries import *


def plot_effect_sankey_percent(
    te,
    se,
    ie,
    de,
    se_decomp=None,
    ie_decomp=None,
    title="Causal decomposition (percent of |TV|)",
):

    se_decomp = se_decomp or {}
    ie_decomp = ie_decomp or {}

    abs_te = abs(te)
    abs_se = abs(se)
    total_lv1 = abs_te + abs_se if abs_te + abs_se > 0 else 1.0

    tv_to_te = abs_te / total_lv1
    tv_to_se = abs_se / total_lv1

    abs_ie = abs(ie)
    abs_de = abs(de)
    total_te_children = abs_ie + abs_de if abs_ie + abs_de > 0 else 1.0

    te_to_ie = tv_to_te * (abs_ie / total_te_children)
    te_to_de = tv_to_te * (abs_de / total_te_children)

    se_vals = list(se_decomp.values())
    abs_se_vals = [abs(v) for v in se_vals]
    total_se_child = sum(abs_se_vals) if abs_se_vals else 1.0

    se_child_shares = [tv_to_se * (a / total_se_child) for a in abs_se_vals]

    ie_vals = list(ie_decomp.values())
    abs_ie_vals = [abs(v) for v in ie_vals]
    total_ie_child = sum(abs_ie_vals) if abs_ie_vals else 1.0

    ie_child_shares = [te_to_ie * (a / total_ie_child) for a in abs_ie_vals]

    flows = []
    flows.append(("TV", "TE", tv_to_te, te))
    flows.append(("TV", "SE", tv_to_se, se))
    flows.append(("TE", "IE", te_to_ie, ie))  # I'm using ie reverse
    flows.append(("TE", "DE", te_to_de, de))

    for (z, eff), share in zip(se_decomp.items(), se_child_shares):
        flows.append(("SE", f"SE[{z}]", share, eff))

    for (w, eff), share in zip(ie_decomp.items(), ie_child_shares):
        flows.append(("IE", f"IE[{w}]", share, eff))

    labels = sorted({s for s, _, _, _ in flows} | {t for _, t, _, _ in flows})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    sources, targets, values, link_labels, link_colors = [], [], [], [], []

    for src, tgt, share, eff in flows:
        if share == 0:
            continue

        sources.append(label_to_idx[src])
        targets.append(label_to_idx[tgt])
        values.append(share * 100)  # percentage

        sign = "positive" if eff >= 0 else "negative"
        link_labels.append(
            f"{src} → {tgt}<br>"
            f"effect = {eff:+.4f}<br>"
            f"share ≈ {share * 100:.1f}% of |TV| ({sign})"
        )

        if eff >= 0:
            link_colors.append("rgba(120,180,120,0.7)")  # green
        else:
            link_colors.append("rgba(220,90,90,0.8)")  # red

    node_colors = ["rgba(230,230,230,1)" for _ in labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=25,
                    label=labels,
                    color=node_colors,
                    line=dict(width=0.5),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    label=link_labels,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=title,
        font_size=14,
    )
    fig.show(renderer="browser")


#####################################################


def plot_xspecific_sankey_percent(
    te_x,
    se_x,
    ie_x,
    de_x,
    se_decomp_x=None,
    ie_decomp_x=None,
    x_label="x*",
    exposure_name="Race",
    outcome_name="Income",
    title=None,
):

    se_decomp_x = se_decomp_x or {}
    ie_decomp_x = ie_decomp_x or {}

    if title is None:
        title = (
            f"{outcome_name}: x-specific decomposition for {exposure_name}={x_label}"
        )

    # 1
    abs_te = abs(te_x)
    abs_se = abs(se_x)
    total_lv1 = abs_te + abs_se if abs_te + abs_se > 0 else 1.0

    tv_to_te = abs_te / total_lv1
    tv_to_se = abs_se / total_lv1

    # 2
    abs_ie = abs(ie_x)
    abs_de = abs(de_x)
    total_te_children = abs_ie + abs_de if abs_ie + abs_de > 0 else 1.0

    te_to_ie = tv_to_te * (abs_ie / total_te_children)
    te_to_de = tv_to_te * (abs_de / total_te_children)

    # 3
    se_vals = list(se_decomp_x.values())
    abs_se_vals = [abs(v) for v in se_vals]
    total_se_child = sum(abs_se_vals) if abs_se_vals else 1.0
    se_child_shares = [tv_to_se * (a / total_se_child) for a in abs_se_vals]

    # 4
    ie_vals = list(ie_decomp_x.values())
    abs_ie_vals = [abs(v) for v in ie_vals]
    total_ie_child = sum(abs_ie_vals) if abs_ie_vals else 1.0
    ie_child_shares = [te_to_ie * (a / total_ie_child) for a in abs_ie_vals]

    tv_label = f"TV | X={x_label}"
    flows = []
    flows.append((tv_label, "TE_x", tv_to_te, te_x, None))
    flows.append((tv_label, "SE_x", tv_to_se, se_x, None))

    de_path = f"{exposure_name}→{outcome_name}"
    flows.append(("TE_x", "IE_x", te_to_ie, ie_x, None))
    flows.append(("TE_x", "DE_x", te_to_de, de_x, de_path))

    # SE_x
    for (z, eff), share in zip(se_decomp_x.items(), se_child_shares):
        z_path = f"{exposure_name}↔{z}→{outcome_name}"
        flows.append(("SE_x", f"SE_x[{z}]", share, eff, z_path))

    # IE_x
    for (w, eff), share in zip(ie_decomp_x.items(), ie_child_shares):
        w_path = f"{exposure_name}→{w}→{outcome_name}"
        flows.append(("IE_x", f"IE_x[{w}]", share, eff, w_path))

    labels = sorted({s for s, _, _, _, _ in flows} | {t for _, t, _, _, _ in flows})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    sources, targets, values, link_labels, link_colors = [], [], [], [], []

    for src, tgt, share, eff, path in flows:
        if share == 0:
            continue

        sources.append(label_to_idx[src])
        targets.append(label_to_idx[tgt])
        values.append(share * 100)  # percent of |TV_x|

        sign = "positive" if eff >= 0 else "negative"
        label = (
            f"{src} → {tgt}<br>"
            f"effect_x = {eff:+.4f}<br>"
            f"share ≈ {share * 100:.1f}% of |TV_x| ({sign})"
        )
        if path is not None:
            label += f"<br>path: {path}"

        link_labels.append(label)
        link_colors.append(
            "rgba(120,180,120,0.7)" if eff >= 0 else "rgba(220,90,90,0.8)"
        )

    node_colors = ["rgba(230,230,230,1)" for _ in labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=25,
                    label=labels,
                    color=node_colors,
                    line=dict(width=0.5),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    label=link_labels,
                ),
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=14)
    fig.show()


#####################################################


def plot_z_specific_sankey(
    z_tuple, z_DE, z_IE, exposure_name="Race", outcome_name="Income"
):
    """
    Sankey for a SINGLE z-profile:
        TE(z) -> DE(z)
        TE(z) -> IE(z)
    where TE(z) = DE(z) + IE(z)
    """

    z_label = ", ".join(map(str, z_tuple))
    te_z = z_DE + z_IE

    abs_de = abs(z_DE)
    abs_ie = abs(z_IE)
    total = abs_de + abs_ie if abs_de + abs_ie > 0 else 1.0

    share_de = abs_de / total
    share_ie = abs_ie / total

    flows = [
        ("TE(z)", "DE(z)", share_de, z_DE),
        ("TE(z)", "IE(z)", share_ie, z_IE),
    ]
    labels = sorted({s for s, _, _, _ in flows} | {t for _, t, _, _ in flows})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    sources, targets, values, link_labels, link_colors = [], [], [], [], []

    for src, tgt, share, eff in flows:
        sign = "positive" if eff >= 0 else "negative"
        label = (
            f"{src} → {tgt}<br>"
            f"effect = {eff:+.4f}<br>"
            f"share = {share * 100:.1f}% of |TE(z)| ({sign})"
        )

        sources.append(label_to_idx[src])
        targets.append(label_to_idx[tgt])
        values.append(share * 100)
        link_labels.append(label)
        link_colors.append(
            "rgba(120,180,120,0.7)" if eff >= 0 else "rgba(220,90,90,0.7)"
        )

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels, pad=20, thickness=20, color="rgba(230,230,230,1)"
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=link_labels,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"z-specific effect decomposition: Z = ({z_label})",
        font_size=14,
    )

    fig.show()


#####################################################


def plot_z_specific_decomposition(effects_z, top_k, tol=1e-12):

    rows = []
    for z_tuple, effs in effects_z.items():
        z_TE = effs["z_TE"]
        z_DE = effs["z_DE"]
        z_IE = effs["z_IE"]

        # keep zc combinations with at least one non-zero between z-DE and z-IE
        if (abs(z_DE) > tol) or (abs(z_IE) > tol):
            rows.append(
                {
                    "Z_tuple": z_tuple,
                    "Z_label": ", ".join(map(str, z_tuple)),
                    "z_TE": z_TE,
                    "z_DE": z_DE,
                    "z_IE": z_IE,
                }
            )

    if not rows:
        print("All z-specific effects are (numerically) zero.")
        return

    df_z = pd.DataFrame(rows)
    df_z["abs_TE"] = df_z["z_TE"].abs()

    # if more than top_k, keep top_k by |TE|
    if len(df_z) > top_k:
        df_z = df_z.sort_values("abs_TE", ascending=False).head(top_k)

    # order by TE for plotting
    df_z = df_z.sort_values("z_TE", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=df_z["Z_label"],
            x=df_z["z_DE"],
            name="z-DE (direct)",
            orientation="h",
        )
    )

    fig.add_trace(
        go.Bar(
            y=df_z["Z_label"],
            x=df_z["z_IE"],
            name="z-IE (indirect)",
            orientation="h",
        )
    )

    fig.update_layout(
        barmode="relative",
        title="z-specific decomposition: TE(z) = DE(z) + IE(z)",
        xaxis_title="Effect size",
        yaxis_title="Z profile",
        bargap=0.2,
    )

    fig.add_vline(x=0, line_width=1, line_dash="dash")

    fig.show()
