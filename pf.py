import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from collections import defaultdict
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# -------------------------
# 1) Graph
# -------------------------
def build_drive_graph(place_name: str) -> nx.DiGraph:
    """
    OSMnx drive network -> DiGraph
    - MultiDiGraph parallel edges reduced: keep min 'length' for each (u,v)
    """
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.distance.add_edge_lengths(G)

    Gu = nx.DiGraph()
    Gu.graph.update(G.graph)
    Gu.graph["crs"] = G.graph["crs"]

    for u, v, k, data in G.edges(keys=True, data=True):
        w = data.get("length", np.nan)
        if not np.isfinite(w):
            continue

        if Gu.has_edge(u, v):
            if w < Gu[u][v]["length"]:
                Gu[u][v].update(data)
                Gu[u][v]["length"] = w
        else:
            Gu.add_edge(u, v, **data)
            Gu[u][v]["length"] = w

    return Gu


# -------------------------
# 2) OD sampling
# -------------------------
def sample_od_pairs(G: nx.DiGraph, n_od: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes)
    # replace=False => s!=t 보장
    return [tuple(rng.choice(nodes, 2, replace=False)) for _ in range(n_od)]


# -------------------------
# 3) OD-level forcedness
# -------------------------
def pair_forcedness_ln_status(G: nx.DiGraph, s, t, weight: str = "length"):
    """
    status:
      - 'ok'      : path1, path2 both exist
      - 'no_alt'  : path1 exists, path2 does not
      - 'no_path' : path1 does not exist / bad node
      - 'invalid' : numeric/logic issue (e.g., L1<=0)
    """
    try:
        gen = nx.shortest_simple_paths(G, s, t, weight=weight)

        path1 = next(gen)
        L1 = nx.path_weight(G, path1, weight)

        try:
            path2 = next(gen)
            L2 = nx.path_weight(G, path2, weight)

            if (np.isfinite(L1) and np.isfinite(L2) and L1 > 0 and L2 >= L1):
                return {"status": "ok", "F_ln": float(np.log(L2 / L1)), "path1": path1, "L1": L1, "L2": L2}
            else:
                return {"status": "invalid", "F_ln": np.nan, "path1": path1, "L1": L1, "L2": L2}

        except StopIteration:
            # path1 exists but no path2
            return {"status": "no_alt", "F_ln": np.nan, "path1": path1, "L1": L1, "L2": np.nan}

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"status": "no_path", "F_ln": np.nan, "path1": [], "L1": np.nan, "L2": np.nan}


# -------------------------
# 4) Tables
# -------------------------
def compute_tables(
    G: nx.DiGraph,
    od_pairs,
    weight: str = "length",
):
    """
    Returns:
      df_edge: edge-level table
      df_od  : OD-level table (for debugging/summary)
    """

    edge_forcedness_vals = defaultdict(list)
    edge_path1_count = defaultdict(int)
    edge_no_alt_count = defaultdict(int)

    od_rows = []

    for s, t in od_pairs:
        res = pair_forcedness_ln_status(G, s, t, weight=weight)
        status = res["status"]
        path1 = res["path1"]

        od_rows.append({
            "s": s,
            "t": t,
            "status": status,
            "forcedness_ln": res["F_ln"],
            "L1": res["L1"],
            "L2": res["L2"],
            "path_len_edges": (len(path1) - 1) if path1 else 0
        })

        if not path1:
            continue

        for u, v in zip(path1[:-1], path1[1:]):
            edge_path1_count[(u, v)] += 1

            if status == "ok":
                edge_forcedness_vals[(u, v)].append(res["F_ln"])
            elif status == "no_alt":
                edge_no_alt_count[(u, v)] += 1

    # edge-level
    edge_rows = []
    for (u, v), n_path1 in edge_path1_count.items():
        vals = edge_forcedness_vals.get((u, v), [])
        n_no_alt = edge_no_alt_count.get((u, v), 0)

        edge_rows.append({
            "u": u,
            "v": v,
            "forcedness_ln_mean": float(np.mean(vals)) if vals else np.nan,
            "forcedness_ln_median": float(np.median(vals)) if vals else np.nan,
            "n_forcedness": int(len(vals)),
            "n_path1": int(n_path1),
            "n_no_alt": int(n_no_alt),
            "no_alt_ratio": (n_no_alt / n_path1) if n_path1 > 0 else np.nan
        })

    df_edge = pd.DataFrame(edge_rows)
    df_od = pd.DataFrame(od_rows)

    return df_edge, df_od


# -------------------------
# 5) Minimal summary + save
# -------------------------
def summarize_od_status(df_od: pd.DataFrame):
    status_counts = df_od["status"].value_counts(dropna=False).to_dict()
    reachable = df_od["status"].isin(["ok", "no_alt"]).sum()
    total = len(df_od)
    return {"total": int(total), "reachable": int(reachable), "status_counts": status_counts}


def save_tables(df_edge: pd.DataFrame, df_od: pd.DataFrame, out_dir: str = "outputs"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    df_edge.to_csv(os.path.join(out_dir, "edge_table.csv"), index=False, encoding="utf-8-sig")
    df_od.to_csv(os.path.join(out_dir, "od_table.csv"), index=False, encoding="utf-8-sig")

# -------------------------
# 6) Compare betweenness
# -------------------------

# edge betweenness computation
def compute_edge_betweenness(G, weight="length"):
    """
    Compute edge betweenness centrality.
    Returns DataFrame with columns: [u, v, betweenness]
    """
    eb = nx.edge_betweenness_centrality(
        G,
        weight=weight,
        normalized=True
    )

    return pd.DataFrame(
        [(u, v, val) for (u, v), val in eb.items()],
        columns=["u", "v", "betweenness"]
    )

## compare indices
# pf.py

def compare_forcedness_betweenness(
    df_edge,
    df_btw,
    forced_col="forcedness_ln_median",
    min_path1=3,
    top_k=200
):
    """
    Compare edge-level forcedness with betweenness.
    Returns dict of summary statistics.
    """

    # 안정성 필터
    df = df_edge.copy()
    df = df[
        (df["n_path1"] >= min_path1) &
        np.isfinite(df[forced_col])
    ]

    # merge
    df = df.merge(df_btw, on=["u", "v"], how="inner")
    df = df[np.isfinite(df["betweenness"])]

    # Spearman rank correlation
    rho, pval = spearmanr(df[forced_col], df["betweenness"])

    # Top-K overlap
    topF = set(
        df.sort_values(forced_col, ascending=False)
          .head(top_k)[["u", "v"]]
          .apply(tuple, axis=1)
    )
    topB = set(
        df.sort_values("betweenness", ascending=False)
          .head(top_k)[["u", "v"]]
          .apply(tuple, axis=1)
    )

    inter = len(topF & topB)
    jacc = inter / len(topF | topB) if (topF or topB) else np.nan

    return {
        "n_compared_edges": len(df),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "topk_intersection": int(inter),
        "topk_jaccard": float(jacc),
    }


def plot_rank_rank_forcedness_betweenness(
    df_edge,
    df_btw,
    forced_col="forcedness_ln_median",
    min_path1=3,
    top_k=200,
    s_all=10,
    s_top=25,
    alpha_all=0.35,
    alpha_top=0.85,
    title=None
):
    """
    Rank–rank scatter: Spearman을 '보여주는' 그림.
    - df_edge: edge table (u,v, forcedness_ln_median, n_path1 등)
    - df_btw : betweenness table (u,v, betweenness)
    """

    # 1) filter + merge (compare 함수와 동일 논리)
    df = df_edge.copy()
    df = df[(df["n_path1"] >= min_path1) & np.isfinite(df[forced_col])]
    df = df.merge(df_btw, on=["u", "v"], how="inner")
    df = df[np.isfinite(df["betweenness"])].copy()

    if len(df) == 0:
        raise ValueError("No edges remain after filtering/merge. Check min_path1/forced_col/merge keys.")

    # 2) ranks (1 = highest)
    df["rank_forced"] = df[forced_col].rank(ascending=False, method="average")
    df["rank_btw"] = df["betweenness"].rank(ascending=False, method="average")

    # 3) top-k forcedness flag
    topF = set(
        df.sort_values(forced_col, ascending=False)
          .head(top_k)[["u", "v"]]
          .apply(tuple, axis=1)
    )
    df["is_top_forced"] = df[["u", "v"]].apply(tuple, axis=1).isin(topF)

    # 4) plot
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df.loc[~df["is_top_forced"], "rank_btw"],
        df.loc[~df["is_top_forced"], "rank_forced"],
        s=s_all, alpha=alpha_all, label="All compared edges"
    )
    plt.scatter(
        df.loc[df["is_top_forced"], "rank_btw"],
        df.loc[df["is_top_forced"], "rank_forced"],
        s=s_top, alpha=alpha_top, label=f"Top-{top_k} forcedness edges"
    )
    plt.xlabel("Betweenness rank (1 = highest)")
    plt.ylabel("Forcedness rank (1 = highest)")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df  # 필요하면 df를 반환해 재사용 가능

# 나눠서그리기
def _prepare_cmp_df(df_edge, df_btw, forced_col="forcedness_ln_median", min_path1=3):
    df = df_edge.copy()
    df = df[(df["n_path1"] >= min_path1) & np.isfinite(df[forced_col])]
    df = df.merge(df_btw, on=["u", "v"], how="inner")
    df = df[np.isfinite(df["betweenness"])].copy()

    if len(df) == 0:
        raise ValueError("No edges remain after filtering/merge. Check min_path1/forced_col/merge keys.")
    return df

def plot_rank_rank_global(
    df_edge, df_btw,
    forced_col="forcedness_ln_median",
    min_path1=3,
    title="Rank–Rank (global): forcedness vs betweenness",
    savepath=None
):
    df = _prepare_cmp_df(df_edge, df_btw, forced_col=forced_col, min_path1=min_path1)

    df["rank_forced"] = df[forced_col].rank(ascending=False, method="average")
    df["rank_btw"]    = df["betweenness"].rank(ascending=False, method="average")

    plt.figure(figsize=(6, 5))
    plt.scatter(df["rank_btw"], df["rank_forced"], s=10, alpha=0.35)
    plt.xlabel("Betweenness rank (1 = highest)")
    plt.ylabel("Forcedness rank (1 = highest)")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return df

def plot_rank_rank_topk_only(
    df_edge, df_btw,
    forced_col="forcedness_ln_median",
    min_path1=3,
    top_k=200,
    title=None,
    savepath=None
):
    df = _prepare_cmp_df(df_edge, df_btw, forced_col=forced_col, min_path1=min_path1)

    # ranks (전체에서의 rank를 계산해둬야 top-k의 "위치"가 유지됨)
    df["rank_forced"] = df[forced_col].rank(ascending=False, method="average")
    df["rank_btw"]    = df["betweenness"].rank(ascending=False, method="average")

    df_sorted = df.sort_values(forced_col, ascending=False)
    df_top = df_sorted.head(top_k).copy()

    if title is None:
        title = f"Rank–Rank (top-{top_k} by forcedness), n_path1≥{min_path1}"

    plt.figure(figsize=(6, 5))
    plt.scatter(df_top["rank_btw"], df_top["rank_forced"], s=25, alpha=0.85)
    plt.xlabel("Betweenness rank (1 = highest)")
    plt.ylabel("Forcedness rank (1 = highest)")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return df_top

def plot_rank_rank_global_and_topk(
    df_edge, df_btw,
    forced_col="forcedness_ln_median",
    min_path1=3,
    top_k=200,
    title_global="(a) Global rank–rank",
    title_topk=None,
    savepath=None
):
    df = _prepare_cmp_df(df_edge, df_btw, forced_col=forced_col, min_path1=min_path1)

    df["rank_forced"] = df[forced_col].rank(ascending=False, method="average")
    df["rank_btw"]    = df["betweenness"].rank(ascending=False, method="average")

    df_sorted = df.sort_values(forced_col, ascending=False)
    df_top = df_sorted.head(top_k).copy()

    if title_topk is None:
        title_topk = f"(b) Top-{top_k} forcedness edges"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) global
    axes[0].scatter(df["rank_btw"], df["rank_forced"], s=10, alpha=0.35)
    axes[0].set_xlabel("Betweenness rank (1 = highest)")
    axes[0].set_ylabel("Forcedness rank (1 = highest)")
    axes[0].set_title(title_global)

    # (b) top-k only
    axes[1].scatter(df_top["rank_btw"], df_top["rank_forced"], s=25, alpha=0.85)
    axes[1].set_xlabel("Betweenness rank (1 = highest)")
    axes[1].set_ylabel("Forcedness rank (1 = highest)")
    axes[1].set_title(title_topk)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return df, df_top

def plot_topk_overlap_curve(
    df_edge,
    df_btw,
    forced_col="forcedness_ln_median",
    min_path1=3,
    k_list=None,
    title=None):
    """
    Top-k overlap curve:
      overlap_ratio(k) = |Top-k(forced) ∩ Top-k(btw)| / k
    + Jaccard(k)도 함께 계산해서 DataFrame으로 반환.
    """

    if k_list is None:
        k_list = [20, 50, 100, 150, 200, 300, 500]

    # filter + merge (compare 함수와 동일)
    df = df_edge.copy()
    df = df[(df["n_path1"] >= min_path1) & np.isfinite(df[forced_col])]
    df = df.merge(df_btw, on=["u", "v"], how="inner")
    df = df[np.isfinite(df["betweenness"])].copy()

    if len(df) == 0:
        raise ValueError("No edges remain after filtering/merge. Check min_path1/forced_col/merge keys.")

    # edge id
    df["edge_id"] = df[["u", "v"]].apply(tuple, axis=1)

    forced_sorted = df.sort_values(forced_col, ascending=False)["edge_id"].tolist()
    btw_sorted = df.sort_values("betweenness", ascending=False)["edge_id"].tolist()

    rows = []
    for k in k_list:
        A = set(forced_sorted[:k])
        B = set(btw_sorted[:k])
        inter = len(A & B)
        union = len(A | B)
        rows.append({
            "k": int(k),
            "intersection": int(inter),
            "overlap_ratio": float(inter / k),
            "jaccard": float(inter / union) if union else np.nan
        })

    out = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    plt.plot(out["k"], out["overlap_ratio"], marker="o")
    plt.xlabel("k")
    plt.ylabel("|Top-k(F) ∩ Top-k(B)| / k")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

    return out