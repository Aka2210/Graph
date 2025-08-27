from asyncio import Queue
from collections import defaultdict, deque
from enum import Enum
import math
import random
import networkx as nx
import heapq
from typing import Any, Dict, List, Set, Tuple
import Debug

INF = float("inf")

class OffPA(Enum):
    USER = "dest"
    SAT = "satellite"
    CLOUD = "cloud"
    SRC = "src"
    REGION = "region"
    WEIGHT = "latency"
    
class TVM(Enum):
    WEIGHT = "cost_traffic"
    USER = "dest"

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

def cost_cache(node_attr, size, alpha=1):
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
                thd = Thd_Latency.get(G.nodes[req_r][OffPA.REGION.value], INF)
                if dist_rj <= thd:
                    candidate[j][req_r] = dist_rj
            
        for j in (sats + clouds):
            size_j[j] = sum(G.nodes[r]["req_size"] for r in candidate[j].keys())
        
        j_bar_val = -INF
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
                curr_j_bar_val = -INF
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

def union_graphs(G1: nx.DiGraph, G2: nx.DiGraph) -> nx.DiGraph:
    """合併兩圖，完整保留並合併屬性；G2 的屬性覆蓋同名鍵。"""
    G_union = nx.DiGraph()

    if G1 is None:
        G1 = nx.DiGraph()
    if G2 is None:
        G2 = nx.DiGraph()
    # 先拷貝 G1
    for n, attrs in G1.nodes(data=True):
        G_union.add_node(n, **attrs)
    for u, v, attrs in G1.edges(data=True):
        G_union.add_edge(u, v, **attrs)

    # 合併 G2（屬性覆蓋）
    for n, attrs in G2.nodes(data=True):
        if n in G_union:
            G_union.nodes[n].update(attrs)
        else:
            G_union.add_node(n, **attrs)

    for u, v, attrs in G2.edges(data=True):
        if G_union.has_edge(u, v):
            G_union[u][v].update(attrs)
        else:
            G_union.add_edge(u, v, **attrs)

    return G_union

def PDTA_Density(G: nx.DiGraph, Beta: float, terminals: Set[str]):
    if G.number_of_edges() == 0:
        return INF
    local_terms = set(terminals)
    D_T = 0
    total = 0
    for u, v, d in G.edges(data=True):
        if v in local_terms:
            local_terms.remove(v)
            D_T += 1
        total += d["cost_traffic"]

    if D_T == 0:
        return INF
    else:
        return (total + Beta * D_T) / D_T

def PDTA(level: int, r: str, m: int, terminals: Set[str], G: nx.DiGraph):
    T_return = nx.DiGraph()
    T_terminals = set(terminals)
    d_T_min_return = INF

    if m <= 0 or not T_terminals or level < 1:
        return T_return, d_T_min_return

    if level == 1:
        edges_sorted = sorted(
            [(u, v, d) for u, v, d in G.edges(data=True) if (u == r and v in T_terminals)],
            key=lambda x: x[2]["cost_traffic"],
            reverse=False
        )
        
        T_return.add_node(r, **G.nodes[r])
        for u, v, d in edges_sorted:
            T_return.add_node(v, **G.nodes[v])
            T_return.add_edge(u, v, **d)
        return T_return, d_T_min_return
        
    D_current = set()
    while len(D_current) < m and T_terminals:
        T_min = nx.DiGraph()
        d_T_min = INF
        tmp: nx.DiGraph
        D_min = set()

        for v in G.successors(r):
            for n in range(1, len(T_terminals) + 1):
                if v in T_terminals:
                    T_terminals.remove(v)
                tmp, density = PDTA(level-1, v, n, T_terminals, G)
                tmp.add_node(r, **G.nodes[r])
                tmp.add_node(v, **G.nodes[v])
                tmp.add_edge(r, v, **G[r][v])

                d_tmp = PDTA_Density(tmp, 1, T_terminals) 
                if d_tmp < d_T_min:
                    T_min = tmp
                    d_T_min = d_tmp
            if G.nodes[v]["type"] == "dest":
                T_terminals.add(v)
        D_min = {n for n in T_min.nodes if G.nodes[n].get("type") == "dest"}
        if not D_min:
            return T_return, INF
        D_current |= D_min
        T_terminals -= D_min
        T_return = union_graphs(T_return, T_min)
        d_T_min_return = d_T_min

    return T_return, d_T_min

# paths = {v: reconstruct_path(parent, v) for v in parent}
def reconstruct_path(parent, target):
    path = []
    u = target
    while u is not None:
        path.append(u)
        u = parent.get(u)
    return list(reversed(path))

def TIG_CTIG(G_sequence: List[nx.DiGraph], srcs: List[str], caches: List[str]):
    T = len(G_sequence)
    E_sets = [set(G.edges()) for G in G_sequence]
    TIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph] = {}
    CTIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph] = {}
    TIG_Edges_Map: Dict[Tuple[int, int, int], Dict[str, List[str]]] = {}
    CTIG_Edges_Map: Dict[Tuple[int, int, int], Dict[str, List[str]]] = {}
    
    for idx, si in enumerate(srcs):
        dists: Dict = {}
        parents: Dict = {}
        for t, G in enumerate(G_sequence):
            need = set(caches)
            visited: set = set()
            dist = {si: 0.0}
            parent = {si: None} 
            pq = [(0.0, si)]
            while pq and need:
                d, u = heapq.heappop(pq)
                if u in need:
                    need.remove(u)
                if d != dist.get(u, math.inf):
                    continue
                if u in visited:
                    continue
                else:
                    visited.add(u)

                for v in G.successors(u):
                    w = G[u][v].get(TVM.WEIGHT.value)
                    nd = d + w
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        parent[v] = u
                        heapq.heappush(pq, (nd, v))
                    
            dists[t] = dist
            parents[t] = parent
            
        for i in range(T):
            current_edges = set(E_sets[i])

            sum_cost: Dict[Tuple[str, str], float] = {
                e: float(G_sequence[i][e[0]][e[1]][TVM.WEIGHT.value]) * G_sequence[i].nodes[si]["data_size"] for e in current_edges
            }

            base_attrs: Dict[Tuple[str, str], dict] = {
                e: {k: v for k, v in G_sequence[i][e[0]][e[1]].items() if k != TVM.WEIGHT.value}
                for e in current_edges
            }
            
            sum_cache_cost = {(si, c): 0.0 for c in caches}
            
            sum_storage_cost = {(si, c): 0.0 for c in caches}

            for j in range(i, T):        
                present = E_sets[j]
                current_edges = current_edges & present
                
                sum_cost = {e: sum_cost[e] for e in current_edges}
                base_attrs = {e: base_attrs[e] for e in current_edges}
                
                if not current_edges:
                    break

                for (u, v) in current_edges:
                    sum_cost[(u, v)] += float(G_sequence[j][u][v][TVM.WEIGHT.value]) * G_sequence[j].nodes[si]["data_size"]

                G_i = G_sequence[i]
                TIG_i_j = nx.DiGraph()
                TIG_i_j.add_nodes_from((n, G_i.nodes[n]) for n in G_i.nodes)

                for (u, v) in current_edges:
                    attrs = dict(base_attrs[(u, v)])
                    attrs[TVM.WEIGHT.value] = sum_cost[(u, v)]
                    TIG_i_j.add_edge(u, v, **attrs)
                
                map = {v: reconstruct_path(parent, v) for v in parent}
                for c in caches:
                    d_val = dists[j].get(c, INF)
                    if d_val == INF:
                        continue

                    sum_cache_cost[(si, c)] += d_val * G_sequence[j].nodes[si]["data_size"]
                    sum_storage_cost[(si, c)] += cost_cache(
                        G_sequence[j].nodes[c],
                        G_sequence[j].nodes[si]["data_size"],
                        alpha=1
                    )

                    total_val = sum_cache_cost[(si, c)] + sum_storage_cost[(si, c)]
                    TIG_i_j.add_edge(si, c, **{TVM.WEIGHT.value: total_val})
                 
                    if (idx, i, j) not in TIG_Edges_Map:
                        TIG_Edges_Map[(idx, i, j)] = {}
                    if c in map:
                        TIG_Edges_Map[(idx, i, j)][c] = map[c]

                TIG_Interval[(idx, i, j)] = TIG_i_j
                K = nx.DiGraph()
                K.add_nodes_from((n, dict(attrs)) for n, attrs in TIG_i_j.nodes(data=True))
                for u in K.nodes:
                    dist, paths = nx.single_source_dijkstra(
                        TIG_i_j,
                        source=u,
                        weight=TVM.WEIGHT.value
                    )
                    CTIG_Edges_Map[(u, i, j)] = paths
                    for v, d in dist.items():
                        if u == v:
                            continue
                        K.add_edge(u, v, **{TVM.WEIGHT.value: float(d)})
                CTIG_Interval[(idx, i, j)] = K
                
    return TIG_Interval, CTIG_Interval

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
        dist, paths = nx.single_source_dijkstra(G, source=si, weight=None)

        # 檢查所有 dest 是否可達
        unreachable = [d for d in local_dests if d not in dist]
        if unreachable:
            print(f"[警告] 圖 key={(idx,i,j)}，source={si} 無法到達以下節點: {unreachable}")
            Debug.print_graph(G)

def TSMTA(TIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph], CTIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph]
          , TIG_Edges_Map: Dict[Tuple[int, int], Dict[str, List[str]]], CTIG_Edges_Map: Dict[Tuple[int, int], Dict[str, List[str]]]
          , srcs: List[str], dests: Dict[Tuple[int, int, int], Set[str]], total_time:int):
    T_i_t: Dict[Tuple[int, int], nx.DiGraph] = {}
    PDTA_cache = {}
    check_reachability(CTIG_Interval, srcs, dests)
    while dests:
        T_best = nx.Graph()
        i_best = 1
        t1_best = 1
        t2_best = 1
        T_Density_min = INF
        for idx, si in enumerate(srcs):
            for i in range(total_time):
                for j in range(i, total_time):
                    local_dests = dests.get((idx, i, j), set())
                    G = CTIG_Interval[(idx, i, j)]
                    sig = graph_signature(G)
                    for k in range(1, len(dests.get((idx, i, j), set())) + 1):
                        # k = len(CTIG_Interval[(idx, i, j)].nodes)
                        cache_key = (sig, si, k)
                        if cache_key in PDTA_cache:
                            tmp_k, tmp_min = PDTA_cache[cache_key]
                        else:
                            tmp_k, tmp_min = PDTA(2, si, k, local_dests, CTIG_Interval[(idx, i, j)])
                            PDTA_cache[cache_key] = (tmp_k, tmp_min)
                        # print(tmp_min)
                        if T_Density_min > tmp_min:
                            T_Density_min = tmp_min
                            T_best = tmp_k.copy()
                            i_best = idx
                            t1_best = i
                            t2_best = j
                    
                        # print(f"{idx}, {i}, {j}, {k}")
        remove = [n for n, d in T_best.nodes(data=True) if d["type"] == TVM.USER.value]
        if len(remove) == 0:
            break 
        # print(t1_best, t2_best)
        for i in range(t1_best, t2_best + 1):
            T_i_t[(i_best, i)] = union_graphs(T_i_t.get((i_best, i), None), T_best)
            for j in range(i, t2_best + 1):
                key = (i_best, i, j)
                dests[key] = dests.get(key, set()) - set(remove)
                if not dests[(i_best, i, j)]:
                    del dests[(i_best, i, j)]
                for u, v in T_best.edges():
                    if TIG_Interval[(i_best, i, j)].has_edge(u, v):
                        TIG_Interval[(i_best, i, j)][u][v][TVM.WEIGHT.value] = 0.0
                    if CTIG_Interval[(i_best, i, j)].has_edge(u, v):
                        CTIG_Interval[(i_best, i, j)][u][v][TVM.WEIGHT.value] = 0.0
        
        for i in range(total_time):
            for j in range(i, total_time):     
                print(i, j, dests.get((i_best, i, j), set()))
                
    print("keys:", list(T_i_t.keys()))