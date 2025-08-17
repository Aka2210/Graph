from asyncio import Queue
from collections import defaultdict, deque
from enum import Enum
import math
import random
import networkx as nx
import heapq
from typing import Any, Dict, List, Set, Tuple


class OffPA(Enum):
    USER = "dest"
    SAT = "satellite"
    CLOUD = "cloud"
    SRC = "src"
    REGION = "region"
    WEIGHT = "latency"

def single_source_MBBSP(graph: nx.DiGraph, sources: Set[str]) -> Tuple[dict, dict]:
    bottleneck = {node: 0 for node in graph.nodes}
    prev = {}

    heap = []
    for s in sources:
        bottleneck[s] = float('inf')
        heapq.heappush(heap, (-bottleneck[s], s))

    while heap:
        cur_bw_neg, u = heapq.heappop(heap)
        cur_bw = -cur_bw_neg
        
        for v in graph.successors(u):
            edge_bw = graph[u][v]['bandwidth']
            new_bw = min(cur_bw, edge_bw)
            if new_bw > bottleneck[v]:
                bottleneck[v] = new_bw
                prev[v] = u
                heapq.heappush(heap, (-new_bw, v))

    return bottleneck, prev

def MBBSP_multicast(graph: nx.DiGraph, s: str, destinations: List[str]) -> float:
    connected: Set[str] = {s}
    remaining = set(destinations)
    total_min_bottleneck = float('inf')

    while remaining:
        bottleneck, prev = single_source_MBBSP(graph, connected)

        best_d = max(remaining, key=lambda d: bottleneck[d])
        best_bw = bottleneck[best_d]

        if best_bw == 0:
            raise ValueError("No path found to some destination.")

        remaining.remove(best_d)
        total_min_bottleneck = min(total_min_bottleneck, best_bw)

    return total_min_bottleneck

def LMBBSP_multicast(graph: nx.DiGraph, s: str, destinations: List[str], alpha: float, c: int):
    threshold = max(MBBSP_multicast(graph, s, destinations) * (1 - alpha), graph.nodes[s]["bandwidth"])
    edges_to_keep = [
            (u, v) for u, v, data in graph.edges(data=True)
            if data.get('bandwidth', 0) >= threshold
        ]
    H = graph.edge_subgraph(edges_to_keep).copy()
    
    # 檢查可達性（pred 不含不可達節點）
    try:
        pred, dist = nx.dijkstra_predecessor_and_distance(H, source=s, weight=None)
    except nx.NodeNotFound as e:
        # 這裡理論上不會再出現（前面已檢查），保留以便訊息更友善
        raise nx.NodeNotFound(f"{e}. Check threshold={threshold:.2f} and alpha.") from e
    except nx.NetworkXNoPath as e:
        raise ValueError(
            f"No path from {s} to at least one destination under threshold={threshold:.2f}."
        ) from e

    unreachable = [d for d in destinations if d not in pred]
    if unreachable:
        raise ValueError(
            f"Unreachable destinations from {s} in H: {unreachable} (threshold={threshold:.2f})."
        )
    
    results = []
    
    for _ in range(c):
        pred, dist = nx.dijkstra_predecessor_and_distance(H, source=s, weight=None)

        if any(not pred.get(d) for d in destinations):
            break

        path_edges = set()
        for dest in destinations:
            cur = dest
            while cur != s and pred[cur]:
                p = pred[cur][0]
                path_edges.add((p, cur))
                cur = p

        subG = H.edge_subgraph(path_edges).copy()
        results.append(subG)

        H.remove_edges_from(path_edges)

    return results

def DMTS(time_slots: int, graphs: List[List[nx.DiGraph]]):
    C = {}
    parents = {(0, i): 0 for i in range(len(graphs[0]))}
    dp = {}
    result = []
    for t in range(1, time_slots):
        for i in range(len(graphs[t])):
            for j in range(len(graphs[t - 1])):
                G1 = graphs[t][i]
                G2 = graphs[t - 1][j]
                edges1 = set(G1.edges())
                edges2 = set(G2.edges())

                C[(t, j, i)] = len(edges1.symmetric_difference(edges2))
                
    for t in range(time_slots):
        for c in range(len(graphs[t])):
            dp[(t, c)] = float('inf')
    
    for c in range(len(graphs[0])):
        dp[(0, c)] = 0
        
    for t in range(1, time_slots):
        for i in range(len(graphs[t])):
            for j in range(len(graphs[t - 1])):
                if dp[(t - 1, j)] + C[(t, j, i)] < dp[(t, i)]:
                    dp[(t, i)] = dp[(t - 1, j)] + C[(t, j, i)]
                    parents[(t, i)] = j
    
    if time_slots == 1:
        tmp = float('inf')
        r = None
        for i in range(len(graphs[0])):
            edges = set(graphs[0][i].edges())
            if len(edges) < tmp:
                tmp = len(edges)
                r = graphs[0][i]
        return [r]
    
    R = None
    min_sum = float('inf')
    for c in range(len(graphs[time_slots - 1])):
        if dp[(time_slots - 1, c)] < min_sum:
            min_sum = dp[(time_slots - 1, c)]
            R = c
    
    for t in range(time_slots - 1, -1, -1):
        result.append(graphs[t][R])
        R = parents[(t, R)]
    
    result.reverse()
    return result

def Setting_Starfront_Thd(G: nx.DiGraph, default_thd: float = 50.0):
    rset: Set[Any] = set()
    for _, d in G.nodes(data=True):
        if d["type"] != OffPA.USER.value:
            continue
        r = d[OffPA.REGION.value]
        if r is not None:
            rset.add(r)
            
    # 暫時統一所有區域的Thd，後續再調整
    return {r: default_thd for r in sorted(rset)}

def multi_src_to_one_dest_subgraph(G, srcs, dest, weight="latency", with_attrs=False):
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

def STARFRONT_sequences(graphs: List[nx.DiGraph], Thd_Latency: dict[str, float] = None):
    if not Thd_Latency:
        Thd_Latency = Setting_Starfront_Thd(graphs[0])
        
    res: List[nx.DiGraph] = []
    
    for i in range(len(graphs)):
        res.append(STARFRONT(G=graphs[i], Thd_Latency=Thd_Latency))
    return res
    
def STARFRONT(G: nx.DiGraph, Thd_Latency: dict[str, float]):
    RQ_remain = {n for n, d in G.nodes(data=True) if d["type"] == OffPA.USER.value}
    dests = [n for n, d in G.nodes(data=True) if d["type"] == OffPA.USER.value]
    srcs  = [n for n, d in G.nodes(data=True) if d["type"] == "src"]
    sats  = [n for n, d in G.nodes(data=True) if d["type"] == "satellite"]
    clouds= [n for n, d in G.nodes(data=True) if d["type"] == "cloud"]
    DG = nx.DiGraph()
    
    def CT_dist(DG: nx.DiGraph):
        res = 0
        srcs = [n for n, d in DG.nodes(data=True) if d.get("type") == "src"]

        for src in srcs:
            q = deque([src])
            visited = set()

            while q:
                u = q.popleft()
                if u in visited:
                    continue
                visited.add(u)

                for v in DG.successors(u):
                    q.append(v)
                    data_size = DG.nodes[u].get("data_size", 0.0)
                    cost = DG[u][v].get("cost_traffic", 0.0)
                    res += data_size * cost

        return res
    
    def CT_storage(DG: nx.DiGraph):
        def cost_cache(node_attr, size):
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
            return gamma * f_size

        res = 0.0
        # 總內容大小 (所有 Wk 的和)
        W_total = sum(DG.nodes[n].get("data_size", 0.0) for n, d in DG.nodes(data=True) if d.get("type") == "src")

        for i, attr in DG.nodes(data=True):
            if attr["type"] in ("cloud", "satellite"):
                # 這裡假設 x(i)=1 代表 DG 裡面有這個點
                res += cost_cache(attr, W_total)
        return res
    
    def CT_access(DG: nx.DiGraph):
        res = 0.0
        # 所有用戶請求節點
        users  = [n for n, d in DG.nodes(data=True) if d.get("type") == OffPA.USER.value]

        for r in users:
            q = deque([r])
            visited = set()
            
            while q:
                u = q.popleft()
                if u in visited:
                    continue
                visited.add(u)

                for v in DG.successors(u):  # 只會走到 caches
                    q.append(v)
                    sz = DG.nodes[u].get("req_size", 0.0)
                    cost = DG[u][v].get("cost_traffic", 0.0)
                    res += sz * cost
        return res
        
    def CT(DG: nx.DiGraph):
        if not DG:
            return 0
        
        return CT_dist(DG) + CT_storage(DG) + CT_access(DG)
    cnt = 0
    while RQ_remain:
        size_j = {}
        candidate = defaultdict(dict)   # candidate[j][req] = best_latency(req -> j)
        res_cache = {}                  # 可選：暫存每個 j 的路徑結果，後面 extend DG 會用到

        for j in (sats + clouds):
            # 一次取回所有 req_r→j 的最短路徑與距離
            res = multi_src_to_one_dest_subgraph(
                G,
                srcs=list(RQ_remain),
                dest=j,
                weight=OffPA.WEIGHT.value,
                with_attrs=True,  # 之後 add_nodes_edges_with_attrs 會用到
            )
            res_cache[j] = res

            # res["paths"]: dict[src] = (dist, [src,...,j])
            for req_r, (dist_rj, path) in res["paths"].items():
                thd = Thd_Latency.get(G.nodes[req_r][OffPA.REGION.value], float("inf"))
                if dist_rj <= thd:
                    candidate[j][req_r] = dist_rj
            
        for j in (sats + clouds):
            size_j[j] = sum(G.nodes[r]["req_size"] for r in candidate[j].keys())
        
        j_bar_val = float("-inf")
        j_bar = -1
        new_DG = DG
        for j in (sats + clouds):
            tmp_DG = DG.copy()
            extend = multi_src_to_one_dest_subgraph(G, (list(candidate[j].keys()) + srcs), j, OffPA.WEIGHT.value, True)
            tmp_DG = add_nodes_edges_with_attrs(tmp_DG, G, 
                                        nodes=extend["nodes"],
                                        edges=extend["edges"],
                                        nodes_attr=extend.get("nodes_attr"),
                                        edges_attr=extend.get("edges_attr")
                                        )
            dCT = CT(tmp_DG) - CT(DG)
            if dCT <= 0:
                curr_j_bar_val = float("-inf")
            else:
                curr_j_bar_val = size_j[j] / dCT
                
            if curr_j_bar_val > j_bar_val:
                j_bar_val = curr_j_bar_val
                new_DG = tmp_DG.copy()
                j_bar = j
        new_RQ_remain = RQ_remain - set(candidate[j_bar].keys())
        
        if new_RQ_remain == RQ_remain:
            print(f"{cnt}: {RQ_remain}\n{new_DG}\n{new_RQ_remain}")
            raise ValueError("No path found to some destination.")
        else:
            DG = new_DG
            RQ_remain = new_RQ_remain
        cnt += 1
    return new_DG