# pf.py
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from collections import defaultdict


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
    Gu.add_nodes_from(G.nodes(data=True))

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
