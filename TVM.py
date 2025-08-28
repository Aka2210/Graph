from enum import Enum
import math
import time
import networkx as nx
import heapq
from typing import Dict, List, Set, Tuple
import Algorithm
import PDTA

INF = float("inf")
    
class TVM(Enum):
    WEIGHT = "cost_traffic"
    USER = "dest"
    
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

                G_j = G_sequence[j]
                TIG_i_j = nx.DiGraph()
                TIG_i_j.add_nodes_from((n, G_j.nodes[n]) for n in G_j.nodes)

                for (u, v) in current_edges:
                    attrs = dict(base_attrs[(u, v)])
                    attrs[TVM.WEIGHT.value] = sum_cost[(u, v)]
                    TIG_i_j.add_edge(u, v, **attrs)
                
                map = {v: Algorithm.reconstruct_path(parents[j], v) for v in parents[j]}
                for c in caches:
                    d_val = dists[j].get(c, INF)
                    if d_val == INF:
                        continue

                    sum_cache_cost[(si, c)] += d_val * G_sequence[j].nodes[si]["data_size"]
                    sum_storage_cost[(si, c)] += Algorithm.cost_cache(
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

def TSMTA(TIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph], CTIG_Interval: Dict[Tuple[int, int, int], nx.DiGraph]
          , TIG_Edges_Map: Dict[Tuple[int, int], Dict[str, List[str]]], CTIG_Edges_Map: Dict[Tuple[int, int], Dict[str, List[str]]]
          , srcs: List[str], dests: Dict[Tuple[int, int, int], Set[str]], total_time:int):
    start_time = time.time()
    T_i_t: Dict[Tuple[int, int], nx.DiGraph] = {}
    PDTA_cache = {}
    Algorithm.check_reachability(CTIG_Interval, srcs, dests)
    pdta_calls = 0
    cache_hits = 0
    time_pdta = 0.0
    time_cache = 0.0
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
                    dcount = len(local_dests)
                    G = CTIG_Interval[(idx, i, j)]
                    sig = Algorithm.graph_signature(G)
                    
                    cache_key = (sig, si, dcount)
                    t0 = time.time()
                    if cache_key in PDTA_cache:
                        cache_hits += 1
                        t0 = time.time()
                        tmp_k, tmp_min, records = PDTA_cache[cache_key]
                        time_cache += time.time() - t0
                    else:
                        pdta_calls += 1
                        tmp_k, tmp_min, records = PDTA.PDTA(2, si, dcount, local_dests, G)
                        time_pdta += time.time() - t0
                        PDTA_cache[cache_key] = (tmp_k, tmp_min, records)
                    
                    if tmp_min < T_Density_min:
                        T_Density_min, T_best, i_best, t1_best, t2_best = tmp_min, tmp_k.copy(), idx, i, j
                    
                    sorted_records = sorted(records.items(), key=lambda x: x[0])
                    total_dests = sum(key[1] for key, _ in sorted_records)
                    
                    tmp_k = nx.DiGraph()
                    tmp_k_cnt, ptr = 0, 0
                    
                    for k in range(1, len(dests.get((idx, i, j), set()))):
                        if k > total_dests:
                            break
                        
                        cache_key = (sig, si, k)
                        t0 = time.time()
                        if cache_key in PDTA_cache:
                            cache_hits += 1
                            tmp_k, tmp_min = PDTA_cache[cache_key]
                            time_cache += time.time() - t0
                        else:
                            if tmp_k_cnt >= k:
                                continue
                            while tmp_k_cnt < k:
                                key, G_sub = sorted_records[ptr]
                                tmp_k = Algorithm.union_graphs(tmp_k, G_sub)
                                tmp_k_cnt += key[1]
                                ptr += 1

                            tmp_min = PDTA.PDTA_Density(tmp_k, 1, local_dests) 
                            PDTA_cache[cache_key] = (tmp_k, tmp_min)
                            
                            if tmp_min < T_Density_min:
                                T_Density_min, T_best, i_best, t1_best, t2_best = tmp_min, tmp_k.copy(), idx, i, j
                    
        remove = [n for n, d in T_best.nodes(data=True) if d["type"] == TVM.USER.value]
        if len(remove) == 0:
            break 
        for i in range(t1_best, t2_best + 1):
            T_i_t[(i_best, i)] = Algorithm.union_graphs(T_i_t.get((i_best, i), None), T_best)
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
                
    print("keys:", list(T_i_t.keys()))
    if pdta_calls > 0:
        avg_pdta = time_pdta / pdta_calls
    else:
        avg_pdta = 0

    if cache_hits > 0:
        avg_cache = time_cache / cache_hits
    else:
        avg_cache = 0

    print(f"Average PDTA time: {avg_pdta:.6f}s")
    print(f"Average cache lookup time: {avg_cache:.6f}s")
    if (pdta_calls + cache_hits) > 0 and pdta_calls > 0:
        est_saving = (avg_pdta - avg_cache) * cache_hits
        print(f"Estimated time saved via cache: {est_saving:.4f}s")
        
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f}s")