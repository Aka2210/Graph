from asyncio import Queue
from collections import defaultdict, deque
from enum import Enum
import math
import random
import time
import networkx as nx
import heapq
from typing import Any, Dict, List, Set, Tuple
import Debug

INF = float("inf")

def are_graphs_equal(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    if not isinstance(G1, nx.Graph) or not isinstance(G2, nx.Graph):
        raise TypeError(f"Inputs must be networkx Graphs, got {type(G1)} and {type(G2)}")

    return set(G1.nodes()) == set(G2.nodes()) and set(G1.edges()) == set(G2.edges())
    
def cost_cache(node_attr: dict, size: float, alpha=1):
    """
    計算單一節點的 storage cost
    node_attr: G.nodes[i] 的 dict
    size: 要放的資料大小 (GB)
    """
    d = node_attr.get("d", 0.0)
    z = node_attr.get("z", 1.0)
    gamma = node_attr.get("gamma", 1.0)
    model = node_attr.get("storage_model", "linear")

    if model == "linear":
        f_size = d * size
    elif model == "concave":
        f_size = d * (size ** z)
    else:
        raise ValueError(f"Unknown storage model: {model}")

    # 雲端 γ=1，衛星 γ>=1
    return alpha * gamma * f_size

def multi_src_to_one_dest_subgraph(G, srcs: List[str], dest: str, weight="latency", with_attrs=False):
    """
    在有向圖 G 上，計算多個 source 到同一個 dest 的最短路徑（用反向圖一次跑完）。
    回傳：
      paths:   dict[src] = (dist, [src, ..., dest])                # 只含 reachable 的 src
      nodes:   set，所有路徑上出現過的節點
      edges:   set，所有路徑上的有向邊 (u, v)（方向為原圖方向：src→...→dest）
      nodes_attr / edges_attr（可選）：節點/邊的屬性 dict
    """
    RG = G.reverse(copy=False)    # 反向視圖
    need = set(srcs)
    dist = {dest: 0.0}
    parent = {dest: None}
    pq = [(0.0, dest)]

    paths = {}           # src -> (dist, path_nodes from src..dest)
    nodes_used = set()
    edges_used = set()

    while pq and need:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue

        if u in need:
            # 在 RG 的路徑 u..dest，反過來就是 G 的 src..dest
            path = []
            x = u
            while x is not None:
                path.append(x)
                x = parent.get(x)
            # path 現在是 [src, ..., dest]（已符合原圖方向）
            paths[u] = (d, path)
            need.remove(u)

            # 收集 nodes / edges
            nodes_used.update(path)
            for a, b in zip(path, path[1:]):
                edges_used.add((a, b))

            if not need:
                break

        for v in RG.successors(u):   # 等價於 G.predecessors(u)
            w = RG[u][v].get(weight, 1.0)
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    result = {
        "paths": paths,
        "nodes": nodes_used,
        "edges": edges_used,
    }

    if with_attrs:
        nodes_attr = {n: dict(G.nodes[n]) for n in nodes_used}
        edges_attr = {(u, v): dict(G[u][v]) for (u, v) in edges_used if G.has_edge(u, v)}
        result["nodes_attr"] = nodes_attr
        result["edges_attr"] = edges_attr

    return result


def add_nodes_edges_with_attrs(DG, G, nodes, edges, nodes_attr=None, edges_attr=None):
    """
    將 nodes / edges（以及可選屬性）從 G 複製到 DG。
    - 若 nodes_attr / edges_attr 為 None，則直接從 G 抓當前屬性。
    """
    # 加節點
    for n in nodes:
        if nodes_attr is not None and n in nodes_attr:
            DG.add_node(n, **nodes_attr[n])
        else:
            DG.add_node(n, **G.nodes[n])

    # 加邊
    for (u, v) in edges:
        if edges_attr is not None and (u, v) in edges_attr:
            DG.add_edge(u, v, **edges_attr[(u, v)])
        else:
            DG.add_edge(u, v, **G[u][v])
    
    return DG

def union_graphs(G1: nx.DiGraph, G2: nx.DiGraph) -> nx.DiGraph:
    """合併兩圖，完整保留並合併屬性；G2 的屬性覆蓋同名鍵。"""
    G_union = nx.DiGraph()

    if G1 is None:
        G1 = nx.DiGraph()
    if G2 is None:
        G2 = nx.DiGraph()

    for n, attrs in G1.nodes(data=True):
        G_union.add_node(n, **attrs)
    for u, v, attrs in G1.edges(data=True):
        G_union.add_edge(u, v, **attrs)

    for n, attrs in G2.nodes(data=True):
        if n in G_union:
            G_union.nodes[n].update(attrs)
        else:
            G_union.add_node(n, **attrs)

    for u, v, attrs in G2.edges(data=True):
        if not G_union.has_edge(u, v):
            G_union.add_edge(u, v, **attrs)

    return G_union

# paths = {v: reconstruct_path(parent, v) for v in parent}
def reconstruct_path(parent, target):
    path = []
    u = target
    while u is not None:
        path.append(u)
        u = parent.get(u)
    return list(reversed(path))

def graph_signature(G: nx.DiGraph):
    nodes = frozenset((n, frozenset(attrs.items())) for n, attrs in G.nodes(data=True))
    edges = frozenset((u, v, frozenset(attrs.items())) for u, v, attrs in G.edges(data=True))
    return (nodes, edges)

def check_reachability(CTIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph],
                       srcs: List[str],
                       dests: Dict[Tuple[int, int, int], Set[str]]):
    for (idx, i, j), G in CTIG_Interval.items():
        local_dests = dests.get((idx, i, j), set())
        si = srcs[idx]

        # 只算一次最短路徑
        dist = dict(nx.single_source_shortest_path(G, source=si, cutoff=None))

        # 檢查所有 dest 是否可達
        unreachable = [d for d in local_dests if d not in dist]
        if unreachable:
            print(f"[警告] 圖 key={(idx,i,j)}，source={si} 無法到達以下節點: {unreachable}")
            # Debug.print_graph(G)
            
def dijkstra_min_edges(G, source, weight="weight"):
    dist = {source: (0, 0)}  # (cost, hop數)
    parent = {source: None}
    pq = [(0, 0, source)]    # (cost, hops, node)

    while pq:
        cost, hops, u = heapq.heappop(pq)
        if (cost, hops) > dist[u]:
            continue
        for v in G.successors(u):
            w = G[u][v].get(weight, 1)
            new_cost = cost + w
            new_hops = hops + 1
            if v not in dist or (new_cost, new_hops) < dist[v]:
                dist[v] = (new_cost, new_hops)
                parent[v] = u
                heapq.heappush(pq, (new_cost, new_hops, v))

    # 重建路徑
    paths = {}
    for v in dist:
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        paths[v] = list(reversed(path))
    return dist, paths