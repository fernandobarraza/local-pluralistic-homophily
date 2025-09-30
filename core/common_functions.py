# common_functions.py
# SPDX-License-Identifier: MIT
"""
Shared utilities for pluralistic homophily experiments on graphs with overlapping communities.

Overview
--------
This module provides:
- Community IO helpers.
- Scalar assignment for nodes (s0–s4), with `s1` used in the manuscript pipeline.
- Global assortativity retrieval via igraph and local proxies used in exploratory analyses.
- Structural helpers (BridgeCC, Collective Influence) for NetworkX/igraph/NetworKit.
- Simple community isolation and internal density proxies (legacy experiments).

Dependencies
------------
- numpy, pandas
- igraph, networkx, networkit
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import time
from collections import deque, defaultdict
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    TypeVar,
)

import igraph as ig
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Deprecation helper
# ---------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])

def deprecated(reason: str = "", version: str = "1.0.0") -> Callable[[F], F]:
    """Decorator to mark functions as deprecated."""
    def deco(func: F) -> F:
        msg = f"{func.__name__} is deprecated since v{version}."
        if reason:
            msg += f" {reason}"
        @wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            warnings.simplefilter("default", DeprecationWarning)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return deco

# ---------------------------------------------------------------------
# Dataset and strategy registries (reduced to useful sets)
# ---------------------------------------------------------------------
DATASETS: Dict[str, str] = {
    "so": "StackOverflow",
    "dblp": "DBLP",
    "lj": "LiveJournal",
    "youtube": "YouTube",
    "text": "toy_example",
    # LFR reference profiles
    "LFR_low": "LFR_low",
    "LFR_mid": "LFR_mid",
    "LFR_high": "LFR_high",
}

STRATEGIES: Dict[str, Dict[str, Union[bool, str]]] = {
    # Local pluralistic homophily-driven
    "hv_s1": {"enabled": True, "col": "hv_s1", "marker": "<"},
    "low_hv_s1": {"enabled": True, "col": "hv_s1", "marker": "<"},
    # Structural
    "bridgecc": {"enabled": True, "col": "bridgecc", "marker": "p"},
    # Baseline
    "random": {"enabled": True, "col": None, "marker": "2"},
}
STRATEGY_SPECIALS: List[str] = ["random"]

# ---------------------------------------------------------------------
# IO helpers (communities and graphs)
# ---------------------------------------------------------------------
def load_communities(file_path: Union[str, Path], separator: str = "\t") -> List[List[str]]:
    """
    Load communities from a text file (one community per line).

    Parameters
    ----------
    file_path : str or Path
        Path to the communities file.
    separator : str, default '\\t'
        Token separator in each line.

    Returns
    -------
    list of list of str
        Each inner list contains node IDs (as strings) belonging to a community.
    """
    communities: List[List[str]] = []
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Communities file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            members = re.split(separator, line.strip())
            if members and members[0] != "":
                communities.append(members)
    log.info("Loaded %d communities from %s", len(communities), file_path)
    return communities


def load_node_to_communities_map(filepath: Union[str, Path], delimiter: str = ",") -> Dict[str, List[int]]:
    """
    Load a mapping node_id -> list of community IDs from a CSV-like file.

    Parameters
    ----------
    filepath : str or Path
        Path to file; each row: node_id, comm1, comm2, ...
    delimiter : str, default ','
        Field delimiter.

    Returns
    -------
    dict[str, list[int]]
        Node to communities mapping.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Node-to-communities map not found: {filepath}")
    node_to_comms: Dict[str, List[int]] = {}
    with filepath.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            node_id = str(row[0]).strip()
            comms = [int(x) for x in row[1:] if str(x).strip() != ""]
            node_to_comms[node_id] = comms
    log.info("Loaded node->communities map with %d nodes from %s", len(node_to_comms), filepath)
    return node_to_comms


def load_node_to_communities(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load communities as a DataFrame from CSV-like file (legacy format).

    Each line contains integers: node_id, comm_id1, comm_id2, ...

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    pandas.DataFrame
        Rows of integer IDs. Deprecated format; kept for backward compatibility.
    """
    file_path = Path(file_path)
    rows: List[List[int]] = []
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if line:
                rows.append([int(cell) for cell in line])
    return pd.DataFrame(rows)


def load_graph_igraph(edge_path: Union[str, Path]) -> ig.Graph:
    """
    Load an undirected graph from an edge list in ncol format.

    Parameters
    ----------
    edge_path : str or Path

    Returns
    -------
    igraph.Graph
        Graph with `name` attribute set to stringified vertex index (if absent).
    """
    g = ig.Graph.Read_Ncol(str(edge_path), directed=False)
    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(v.index) for v in g.vs]
    return g


def load_graph_nk(path: Union[str, Path], directed: bool = False) -> nk.Graph:
    """
    Load a NetworKit graph from a tab-separated edgelist with 0-based IDs.

    Parameters
    ----------
    path : str or Path
    directed : bool, default False

    Returns
    -------
    networkit.Graph
    """
    return nk.graphio.readGraph(str(path), nk.Format.EdgeListTabZero, directed=directed)

# ---------------------------------------------------------------------
# Scalar assignment (s0–s4)
# ---------------------------------------------------------------------
def assign_scalar_values(
    graph: ig.Graph,
    method: str,
    communities: Optional[List[List[Union[str, int]]]] = None,
    separator: str = "\t",
    timer: bool = True,
    node_to_comms_dict: Optional[Dict[str, Iterable[int]]] = None,
    out_attr: str = "scalar_value",
) -> List[int]:
    """
    Assign a scalar attribute per node according to the selected method.

    Methods
    -------
    s0 : degree of the node
    s1 : number of unique communities shared with neighbors (recommended for pipeline)
    s2 : number of communities the node belongs to
    s3 : sum over neighbors of |intersection(my_comms, neigh_comms)|
    s4 : number of unique communities shared with neighbors (set-union across neighbors)

    Parameters
    ----------
    graph : igraph.Graph
    method : {'s0','s1','s2','s3','s4'}
    communities : list of list, optional
        Community membership per line, used when node_to_comms_dict is not provided.
    separator : str, default '\\t'
        Only used if `communities` is a raw line-based format (compatibility).
    timer : bool, default True
        If True, log elapsed time.
    node_to_comms_dict : dict[str, Iterable[int]], optional
        Mapping node name -> iterable of community IDs.
    out_attr : str, default 'scalar_value'
        Vertex attribute name used to store the scalar.

    Returns
    -------
    list[int]
        Scalar values in the order of `graph.vs`.
    """
    if method not in {"s0", "s1", "s2", "s3", "s4"}:
        raise ValueError(f"Unsupported method: {method}")

    t0 = time.perf_counter() if timer else None

    # Build node -> set(communities) mapping
    if node_to_comms_dict is None:
        if communities is None:
            raise ValueError("Either node_to_comms_dict or communities must be provided.")
        node_to_communities: Dict[str, List[int]] = {}
        for community_id, community in enumerate(communities):
            for node in community:
                node_id = str(node)
                node_to_communities.setdefault(node_id, []).append(community_id)
        node_to_comms_set: Dict[str, Set[int]] = {k: set(v) for k, v in node_to_communities.items()}
    else:
        node_to_comms_set = {str(k): set(v) for k, v in node_to_comms_dict.items()}

    scalar_values: List[int] = []

    if method == "s0":
        scalar_values = list(graph.degree())
        graph.vs[out_attr] = scalar_values

    elif method in {"s1", "s3", "s4", "s2"}:
        # Pre-index community sets by vertex index
        all_comms_sets: List[Set[int]] = [node_to_comms_set.get(str(v["name"]), set()) for v in graph.vs]

        if method == "s1":
            for v in graph.vs:
                my_comms = all_comms_sets[v.index]
                unique_shared: Set[int] = set()
                for neigh in graph.neighbors(v.index):
                    neigh_comms = all_comms_sets[neigh]
                    unique_shared.update(my_comms & neigh_comms)
                val = len(unique_shared)
                v[out_attr] = val
                scalar_values.append(val)

        elif method == "s2":
            for v in graph.vs:
                val = len(all_comms_sets[v.index])
                v[out_attr] = val
                scalar_values.append(val)

        elif method == "s3":
            for v in graph.vs:
                my_comms = all_comms_sets[v.index]
                total = 0
                for neigh in graph.neighbors(v.index):
                    neigh_comms = all_comms_sets[neigh]
                    total += len(my_comms & neigh_comms)
                v[out_attr] = total
                scalar_values.append(total)

        elif method == "s4":
            for v in graph.vs:
                my_comms = all_comms_sets[v.index]
                unique_shared: Set[int] = set()
                for neigh in graph.neighbors(v.index):
                    neigh_comms = all_comms_sets[neigh]
                    unique_shared.update(my_comms & neigh_comms)
                val = len(unique_shared)
                v[out_attr] = val
                scalar_values.append(val)

    # Basic sanity: non-constant variance (assortativity requires variability)
    if np.var(scalar_values) == 0:
        raise ValueError(
            f"Method '{method}' produced zero variance scalar values. "
            "Check community data or choose another method."
        )

    if timer:
        t1 = time.perf_counter()
        log.info("assign_scalar_values(method=%s) took %.2f s", method, t1 - t0)

    return scalar_values

# ---------------------------------------------------------------------
# Global local-homophily vector (ghv)
# ---------------------------------------------------------------------
def calculate_ghv(graph: ig.Graph, scalar_name: str = "scalar_value") -> Tuple[np.ndarray, float]:
    """
    Compute a degree-normalized local homophily vector per node (ghv),
    whose degree-weighted mean equals global assortativity.

    Parameters
    ----------
    graph : igraph.Graph
    scalar_name : str, default 'scalar_value'

    Returns
    -------
    ghv : np.ndarray
        Per-node value in R.
    h_global : float
        Newman assortativity for the given scalar.
    """
    scalar = np.asarray(graph.vs[scalar_name], dtype=float)
    edges = graph.get_edgelist()

    # Mean/variance over edge endpoints
    edge_scalars: List[float] = []
    for u, v in edges:
        edge_scalars.append(scalar[u])
        edge_scalars.append(scalar[v])
    mu = float(np.mean(edge_scalars))
    sigma2 = float(np.var(edge_scalars))
    if sigma2 == 0.0:
        raise ValueError("Zero variance over edge-endpoint scalar values.")

    ghv = np.zeros(graph.vcount(), dtype=float)
    deg = np.asarray(graph.degree(), dtype=float)
    for v in range(graph.vcount()):
        neighbors = graph.neighbors(v)
        if deg[v] == 0:
            ghv[v] = 0.0
            continue
        ghv[v] = np.sum((scalar[v] - mu) * (scalar[neighbors] - mu)) / (deg[v] * sigma2)

    h_global = float(graph.assortativity(types1=scalar_name, directed=False))
    return ghv, h_global

# ---------------------------------------------------------------------
# Overlap-aware immunization heuristics (Kumar et al. style)
# ---------------------------------------------------------------------
def get_overlapping_nodes(node_comms: Dict[int, Set[str]]) -> List[int]:
    """
    Return nodes that belong to more than one community.

    Parameters
    ----------
    node_comms : dict[int, set[str]]

    Returns
    -------
    list[int]
    """
    return [u for u, comms in node_comms.items() if len(comms) > 1]


def overlap_neighborhood_immunization_nx(G: nx.Graph, overlapping_nodes: Sequence[int]) -> List[int]:
    """
    Rank neighbors of overlapping nodes by degree (NetworkX variant).

    Returns
    -------
    list[int]
        Sorted in descending degree order.
    """
    khoplist: Set[int] = set()
    for u in overlapping_nodes:
        khoplist |= set(G.neighbors(u))
    degs = {v: G.degree(v) for v in khoplist}
    ranked = sorted(degs, key=lambda v: degs[v], reverse=True)
    return ranked


def overlap_neighborhood_immunization_igraph(G: ig.Graph, overlapping_nodes: Sequence[int]) -> List[int]:
    """
    Rank neighbors of overlapping nodes by degree (igraph variant).
    """
    khoplist: Set[int] = set()
    for u in overlapping_nodes:
        khoplist.update(G.neighbors(u))
    degs = G.degree(list(khoplist))
    deg_map = dict(zip(khoplist, degs))
    return sorted(deg_map, key=lambda v: deg_map[v], reverse=True)


def overlap_neighborhood_immunization(Gk: nk.Graph, overlapping_nodes: Sequence[int]) -> List[int]:
    """
    Rank neighbors of overlapping nodes by degree (NetworKit variant).
    """
    khoplist: Set[int] = set()
    for u in overlapping_nodes:
        khoplist.update(Gk.iterNeighbors(u))
    deg_map = {v: Gk.degree(v) for v in khoplist}
    return sorted(deg_map, key=lambda v: deg_map[v], reverse=True)

# ---------------------------------------------------------------------
# Structural metrics (BridgeCC, Collective Influence, approx betweenness)
# ---------------------------------------------------------------------
def compute_bridgecc_nx(G: nx.Graph) -> Dict[int, float]:
    """
    BridgeCC(v) = degree(v) * (1 - local_clustering(v)).
    """
    deg_map = dict(G.degree())
    clustering = nx.clustering(G)
    return {v: deg_map[v] * (1.0 - clustering[v]) for v in G.nodes()}


def compute_bridgecc_igraph(G: ig.Graph) -> Dict[int, float]:
    """
    BridgeCC for igraph.
    """
    degs = G.degree()
    clusts = G.transitivity_local_undirected(vertices=None)
    return {v.index: d * (1.0 - c) for v, d, c in zip(G.vs, degs, clusts)}


def compute_bridgecc(Gk: nk.Graph) -> Dict[int, float]:
    """
    BridgeCC for NetworKit.
    """
    bridgecc: Dict[int, float] = {}
    for u in Gk.iterNodes():
        k_u = Gk.degree(u)
        if k_u < 2:
            bridgecc[u] = 0.0
            continue
        neighbors = list(Gk.iterNeighbors(u))
        triangles = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if Gk.hasEdge(neighbors[i], neighbors[j]):
                    triangles += 1
        clustering = 2 * triangles / (k_u * (k_u - 1))
        bridgecc[u] = k_u * (1.0 - clustering)
    return bridgecc


def compute_collective_influence_nx(G: nx.Graph, radius: int = 2) -> Dict[int, float]:
    """
    Collective Influence at radius l (NetworkX).
    CI_l(v) = (deg(v) - 1) * sum_{u in frontier at distance l} (deg(u) - 1)
    """
    deg_map = dict(G.degree())
    ci: Dict[int, float] = {}
    for v in G.nodes():
        visited = {v}
        queue: deque[Tuple[int, int]] = deque([(v, 0)])
        frontier: Set[int] = set()
        while queue:
            u, d = queue.popleft()
            if d == radius:
                frontier.add(u)
            elif d < radius:
                for w in G.neighbors(u):
                    if w not in visited:
                        visited.add(w)
                        queue.append((w, d + 1))
        ci[v] = (deg_map[v] - 1) * sum(deg_map[u] - 1 for u in frontier)
    return ci


def compute_collective_influence_igraph(G: ig.Graph, radius: int = 2) -> Dict[int, float]:
    """
    Collective Influence at radius l (igraph).
    """
    degs = G.degree()
    ci: Dict[int, float] = {}
    for v in range(G.vcount()):
        visited = {v}
        queue: deque[Tuple[int, int]] = deque([(v, 0)])
        frontier: Set[int] = set()
        while queue:
            u, d = queue.popleft()
            if d == radius:
                frontier.add(u)
            elif d < radius:
                for w in G.neighbors(u):
                    if w not in visited:
                        visited.add(w)
                        queue.append((w, d + 1))
        ci[v] = (degs[v] - 1) * sum(degs[u] - 1 for u in frontier)
    return ci


def compute_collective_influence(Gk: nk.Graph, radius: int = 2) -> Dict[int, float]:
    """
    Collective Influence at radius l (NetworKit).
    """
    ci: Dict[int, float] = {}
    deg_map = {u: Gk.degree(u) for u in Gk.iterNodes()}
    for v in Gk.iterNodes():
        visited = {v}
        queue: deque[Tuple[int, int]] = deque([(v, 0)])
        frontier: Set[int] = set()
        while queue:
            u, d = queue.popleft()
            if d == radius:
                frontier.add(u)
            elif d < radius:
                for w in Gk.iterNeighbors(u):
                    if w not in visited:
                        visited.add(w)
                        queue.append((w, d + 1))
        ci[v] = (deg_map[v] - 1) * sum(deg_map[u] - 1 for u in frontier)
    return ci


def compute_approximate_betweenness(G: nx.Graph, epsilon: float = 0.01) -> Dict[int, float]:
    """
    Approximate betweenness centrality via NetworKit on a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph
    epsilon : float, default 0.01

    Returns
    -------
    dict[int, float]
    """
    G_nk = nk.nxadapter.nx2nk(G)
    abt = nk.centrality.ApproxBetweenness(G_nk, epsilon=epsilon)
    abt.run()
    scores = abt.scores()
    nodes = list(G.nodes())
    return {nodes[i]: scores[i] for i in range(len(nodes))}

# ---------------------------------------------------------------------
# Community isolation and internal density (legacy proxies)
# ---------------------------------------------------------------------
def community_isolation(G: Union[nk.Graph, ig.Graph], communities: Sequence[Sequence[int]]) -> float:
    """
    Average internal-isolation proxy for a set of communities.

    Parameters
    ----------
    G : networkit.Graph or igraph.Graph
    communities : list of list of int

    Returns
    -------
    float
        Mean isolation across communities with size >= 2.
    """
    if isinstance(G, nk.Graph):
        return _community_isolation_networkit(G, communities)
    elif isinstance(G, ig.Graph):
        return _community_isolation_igraph(G, communities)
    else:
        raise TypeError(f"Unsupported graph type: {type(G)}")


def _community_isolation_networkit(Gk: nk.Graph, communities: Sequence[Sequence[int]]) -> float:
    isolation: List[float] = []
    for comm in communities:
        comm_nodes = [int(n) for n in comm if Gk.hasNode(int(n))]
        if len(comm_nodes) < 2:
            continue
        internal = 0
        external = 0
        comm_nodes_set = set(comm_nodes)
        for u in comm_nodes:
            for v in Gk.iterNeighbors(u):
                if v in comm_nodes_set:
                    internal += 1
                else:
                    external += 1
        internal //= 2
        total = internal + external
        iso = internal / total if total > 0 else 0.0
        isolation.append(iso)
    return float(np.mean(isolation)) if isolation else 0.0


def _community_isolation_igraph(G: ig.Graph, communities: Sequence[Sequence[int]]) -> float:
    isolation: List[float] = []
    node_set = set(G.vs.indices)
    for comm in communities:
        comm_nodes = [int(n) for n in comm if int(n) in node_set]
        if len(comm_nodes) < 2:
            continue
        comm_nodes_set = set(comm_nodes)
        internal = 0
        external = 0
        for u in comm_nodes:
            for v in G.neighbors(u):
                if v in comm_nodes_set:
                    internal += 1
                else:
                    external += 1
        internal //= 2
        total = internal + external
        iso = internal / total if total > 0 else 0.0
        isolation.append(iso)
    return float(np.mean(isolation)) if isolation else 0.0


def internal_edge_density(Gk: nk.Graph, communities: Sequence[Sequence[int]]) -> float:
    """
    Mean internal edge density across communities (NetworKit).
    """
    densities: List[float] = []
    for comm in communities:
        comm_nodes = [int(n) for n in comm if Gk.hasNode(int(n))]
        n = len(comm_nodes)
        if n < 2:
            continue
        internal_edges = 0
        comm_nodes_set = set(comm_nodes)
        for u in comm_nodes:
            for v in Gk.iterNeighbors(u):
                if v in comm_nodes_set and u < v:
                    internal_edges += 1
        max_possible = n * (n - 1) // 2
        dens = internal_edges / max_possible if max_possible > 0 else 0.0
        densities.append(dens)
    return float(np.mean(densities)) if densities else 0.0

# ---------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------
def get_node_order(
    strategy_key: str,
    df_metrics: pd.DataFrame,
    node_list: Sequence[int],
    reverse: bool = False,
) -> Optional[List[int]]:
    """
    Return a node order according to the provided strategy.

    Parameters
    ----------
    strategy_key : str
    df_metrics : pandas.DataFrame
        Must contain a column matching STRATEGIES[strategy_key]['col'] if not 'random'.
    node_list : sequence of int
    reverse : bool, default False
        Passed to sort_values(ascending=reverse).

    Returns
    -------
    list[int] or None
        Node order or None if required column is missing.
    """
    if strategy_key == "random":
        return random.sample(list(node_list), len(node_list))
    if strategy_key not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {strategy_key}")
    col = STRATEGIES[strategy_key]["col"]
    if not col:
        raise ValueError(f"Strategy {strategy_key} requires a metric column.")
    if str(col) not in df_metrics.columns:
        log.warning("Column '%s' not found in metrics DataFrame.", col)
        return None
    df_sub = df_metrics[df_metrics["Node ID"].astype(int).isin(node_list)].copy()
    df_sub = df_sub.sort_values(by=str(col), ascending=reverse)
    return df_sub["Node ID"].astype(int).tolist()

# ---------------------------------------------------------------------
# Legacy CLI parsers (kept for compatibility; recommend using argparse per-script)
# ---------------------------------------------------------------------
@deprecated(reason="Use per-script argparse with explicit defaults.", version="1.0.0")
def parse_pipeline_args2(default_dataset: str = "so", default_eq: str = "", default_k: int = 10) -> str:
    """
    Minimal CLI for legacy pipeline; returns dataset key.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        help=f"Dataset key to process. Available: {', '.join(DATASETS.keys())}")
    parser.add_argument("--eq", type=str, default=default_eq, help="Equation version flag.")
    parser.add_argument("--k", type=int, default=default_k, help="Number of seeds/nodes.")
    args = parser.parse_args()
    dataset = args.dataset
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid: {', '.join(DATASETS.keys())}")
    log.info("Dataset selected: %s (%s)", dataset, DATASETS[dataset])
    return dataset


@deprecated(reason="Use per-script argparse with explicit defaults.", version="1.0.0")
def parse_pipeline_args(
    default_dataset: str = "so",
    default_eq: str = "",
    default_k: int = 10,
    default_mode: str = "both",
    default_base_dir: str = ".",
    default_runs: int = 20,
):
    """
    Legacy CLI parser kept for backward compatibility.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        help=f"Dataset key. Available: {', '.join(DATASETS.keys())}")
    parser.add_argument("--eq", type=str, default=default_eq, help="Equation version flag.")
    parser.add_argument("--k", type=int, default=default_k, help="Number of seeds/nodes.")
    parser.add_argument("--mode", type=str, default=default_mode, choices=["calc", "plot", "both"])
    parser.add_argument("--base_dir", type=str, default=default_base_dir)
    parser.add_argument("--runs", type=int, default=default_runs)
    args = parser.parse_args()
    if args.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Valid: {', '.join(DATASETS.keys())}")
    log.info("Dataset selected: %s (%s)", args.dataset, DATASETS[args.dataset])
    return args
