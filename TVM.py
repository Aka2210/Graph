from enum import Enum
import math
import time
import networkx as nx
import heapq
import Algorithm
import PDTA
import main

INF = float("inf")
    
class TVM(Enum):
    WEIGHT = "cost_traffic"
    USER = "dest"
    BETA = 1
    ALPHA = 1
    
def TIG_CTIG(G_sequence: list[nx.DiGraph], srcs: list[str], caches: list[str]):
    T = len(G_sequence)
    E_sets = [set(G.edges()) for G in G_sequence]
    TIG_Interval: dict[tuple[int, int, int], nx.DiGraph] = {}
    CTIG_Interval: dict[tuple[int, int, int], nx.DiGraph] = {}
    TIG_Edges_Map: dict[tuple[int, int, int], dict[str, list[str]]] = {}
    CTIG_Edges_Map: dict[tuple[int, int, int], dict[str, list[str]]] = {}
    
    for idx, si in enumerate(srcs):
        dists: dict = {}
        parents: dict = {}

        for t, G in enumerate(G_sequence):
            dist, paths = Algorithm.dijkstra_min_edges(G, source=si, weight=TVM.WEIGHT.value)

            dists[t] = {v: cost for v, (cost, hops) in dist.items()}

            parent = {}
            for v, path in paths.items():
                if len(path) >= 2:
                    parent[v] = path[-2]
                else:
                    parent[v] = None
            parents[t] = parent
            
        for i in range(T):
            current_edges = set(E_sets[i])
            # print(len(current_edges), i, "start")
            sum_cost: dict[tuple[str, str], float] = {
                e: float(G_sequence[i][e[0]][e[1]][TVM.WEIGHT.value]) * G_sequence[i].nodes[si]["data_size"] for e in current_edges
            }

            base_attrs: dict[tuple[str, str], dict] = {
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
                # print(len(current_edges), j)
                # if not current_edges:
                #     break

                for (u, v) in sorted(current_edges, key=lambda e: (str(e[0]), str(e[1]))):
                    sum_cost[(u, v)] += float(G_sequence[j][u][v][TVM.WEIGHT.value]) * G_sequence[j].nodes[si]["data_size"]

                G_j = G_sequence[j]
                TIG_i_j = nx.DiGraph()
                # TIG_i_j.add_nodes_from((n, G_j.nodes[n]) for n in G_j.nodes)
                TIG_i_j.add_nodes_from((n, dict(attrs)) for n, attrs in G_j.nodes(data=True))

                for (u, v) in sorted(current_edges, key=lambda e: (str(e[0]), str(e[1]))):
                    attrs = dict(base_attrs[(u, v)])
                    attrs[TVM.WEIGHT.value] = sum_cost[(u, v)] / (j - i + 1)
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

                    total_val = (sum_cache_cost[(si, c)] + sum_storage_cost[(si, c)]) / (j - i + 1)
                    TIG_i_j.add_edge(si, c, **{TVM.WEIGHT.value: total_val, "virtual": True})
                 
                    if (idx, j) not in TIG_Edges_Map:
                        TIG_Edges_Map[(idx, j)] = {}
                    if c in map:
                        TIG_Edges_Map[(idx, j)][c] = map[c]

                TIG_Interval[(idx, i, j)] = TIG_i_j
                K = nx.DiGraph()
                K.add_nodes_from((n, dict(attrs)) for n, attrs in TIG_i_j.nodes(data=True))
                for u in sorted(K.nodes(), key=str):
                    dist, paths = Algorithm.dijkstra_min_edges(
                        TIG_i_j,
                        source=u,
                        weight=TVM.WEIGHT.value
                    )
                    CTIG_Edges_Map[(u, i, j)] = paths
                    for v, (cost, hops) in sorted(dist.items(), key=lambda kv: str(kv[0])):
                        if u == v:
                            continue
                        K.add_edge(u, v, **{TVM.WEIGHT.value: float(cost), "virtual": True})
                CTIG_Interval[(idx, i, j)] = K
                
    return TIG_Interval, CTIG_Interval, TIG_Edges_Map, CTIG_Edges_Map

def expand_virtual_edges(T_i_t: dict[tuple[int, int], nx.DiGraph], TIG_Interval: dict[tuple[int, int, int], nx.DiGraph], TIG_Edges_Map: dict[tuple[int, int], dict[str, list[str]]], srcs: list[str], caches: list[str], total_time:int):
    for idx, si in enumerate(srcs):
        for t in range(total_time):
            for v in caches:
                if not T_i_t[(idx, t)].has_edge(si, v):
                    continue
                if not T_i_t[(idx, t)][si][v]["virtual"]:
                    continue
                
                key = (idx, t)
                if key not in TIG_Edges_Map:
                    continue
                
                if v not in TIG_Edges_Map[key]:
                    continue
                
                real_path = TIG_Edges_Map[key][v]
                print(f"Virtual edge ({si}->{v}) 展開路徑: {real_path}")
                
                T_i_t[(idx, t)].remove_edge(si, v)
                for x, y in zip(real_path, real_path[1:]):
                    if TIG_Interval[(idx, t, t)].has_edge(x, y):
                        edge_attr = TIG_Interval[(idx, t, t)][x][y]
                        T_i_t[(idx, t)].add_edge(x, y, **edge_attr)
                    else:
                        raise KeyError(
                            f"❌ Edge ({x} -> {y}) not found in TIG_Interval[{idx}, {t}, {t}]"
                        )

def TSMTA(TIG: dict[tuple[int, int, int], nx.DiGraph], CTIG: dict[tuple[int, int, int], nx.DiGraph]
          , TIG_Edges_Map: dict[tuple[int, int], dict[str, list[str]]], CTIG_Edges_Map: dict[tuple[int, int, int], dict[str, list[str]]]
          , srcs: list[str], caches: list[str], dests: dict[tuple[int, int, int], set[str]], total_time:int, node_attr_map: dict=None):
    T_i_t: dict[tuple[int, int], nx.DiGraph] = {}
    PDTA_cache = {}
    Choosing_cache = {}
    # Algorithm.check_reachability(CTIG_Interval, srcs, dests)
    pdta_calls = 0
    cache_hits = 0
    time_pdta = 0.0
    time_cache = 0.0
    TIG_Interval = TIG_Interval = {k: v.copy() for k, v in TIG.items()}
    CTIG_Interval = TIG_Interval = {k: v.copy() for k, v in CTIG.items()}
    while dests:
        T_best = nx.Graph()
        i_best = 1
        t1_best = 1
        t2_best = 1
        T_Density_min = INF
        
        for idx, si in enumerate(srcs):
            for i in range(total_time):
                local_dests = dests.get((idx, i, i), set())
                for j in range(i, total_time):
                    local_dests = (local_dests & dests.get((idx, j, j), set()))
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
                    
                    for k in range(1, len(local_dests)):
                        if k > total_dests:
                            break
                        
                        cache_key = (sig, si, k)
                        t0 = time.time()
                        if cache_key in Choosing_cache:
                            cache_hits += 1
                            tmp_k, tmp_min = Choosing_cache[cache_key]
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
                            Choosing_cache[cache_key] = (tmp_k, tmp_min)
                            
                            if tmp_min < T_Density_min:
                                T_Density_min, T_best, i_best, t1_best, t2_best = tmp_min, tmp_k.copy(), idx, i, j
                    
        remove = [n for n, d in T_best.nodes(data=True) if d["type"] == TVM.USER.value]
        if len(remove) == 0:
            break 
        # print(t1_best, t2_best)
        edges_to_process = list(T_best.edges())
        for u, v in edges_to_process:
            paths_dict = CTIG_Edges_Map.get((u, t1_best, t2_best), [])
            if v not in paths_dict:
                continue
            real_path = paths_dict[v]

            if T_best.has_edge(u, v):
                T_best.remove_edge(u, v)

            for x, y in zip(real_path, real_path[1:]):
                if x not in T_best:
                    T_best.add_node(x, **node_attr_map.get(x, {}))
                if y not in T_best:
                    T_best.add_node(y, **node_attr_map.get(y, {}))

                if TIG_Interval[(i_best, t1_best, t2_best)].has_edge(x, y):
                    edge_attr = TIG_Interval[(i_best, t1_best, t2_best)][x][y]
                    T_best.add_edge(x, y, **edge_attr)
                else:
                    raise KeyError(
                        f"❌ Edge ({x} -> {y}) not found in TIG_Interval[{i_best}, {t1_best}, {t2_best}]"
                    )
                    
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
                        
    # expand_virtual_edges(T_i_t=T_i_t, TIG_Interval=TIG_Interval, TIG_Edges_Map=TIG_Edges_Map, srcs=srcs, caches=caches, total_time=total_time)

    # if pdta_calls > 0:
    #     avg_pdta = time_pdta / pdta_calls
    # else:
    #     avg_pdta = 0

    # if cache_hits > 0:
    #     avg_cache = time_cache / cache_hits
    # else:
    #     avg_cache = 0
    
    # print(f"Average PDTA time: {avg_pdta:.6f}s")
    # print(f"Average cache lookup time: {avg_cache:.6f}s")
    # if (pdta_calls + cache_hits) > 0 and pdta_calls > 0:
    #     est_saving = (avg_pdta - avg_cache) * cache_hits
    #     print(f"Estimated time saved via cache: {est_saving:.4f}s")
            
    return T_i_t

def BC(T_i_t: dict[tuple[int, int], nx.DiGraph], src_nodes: list[str], total_time: int):
    total_cost = 0.0
    for idx, si in enumerate(src_nodes):
        for t in range(total_time):
            # s: set = set()
            size = T_i_t[(idx, t)].nodes[si].get("data_size", 0.0)
            G = T_i_t.get((idx, t))
            if G is None:
                continue
            for u, v, attrs in G.edges(data=True):
                # s.add((u, v))
                total_cost += attrs.get("cost_traffic", 0.0) * size
            # print(s)
    return total_cost 

def CC(T_i_t: dict[tuple[int, int], nx.DiGraph], src_nodes: list[str], caches: list[str], total_time: int, alpha=1):
    total_cost = 0.0
    for idx, si in enumerate(src_nodes):
        for t in range(total_time):
            G = T_i_t.get((idx, t))
            if G is None:
                continue

            size = G.nodes[si].get("data_size", 0.0)

            for c in caches:
                if c not in G.nodes:
                    continue
                node_attr = G.nodes[c]
                total_cost += Algorithm.cost_cache(node_attr, size, alpha)
    return total_cost

def RC(T_i_t: dict[tuple[int, int], nx.DiGraph], src_nodes: list[str], total_time: int, beta=1):
    total_cost = 0
    for idx, si in enumerate(src_nodes):
        for t in range(total_time - 1):
            G1 = T_i_t.get((idx, t))
            G2 = T_i_t.get((idx, t + 1))
            
            edges1 = set(G1.edges())
            edges2 = set(G2.edges())

            total_cost += len(edges1.symmetric_difference(edges2))
    return total_cost * beta

def evaluate_algorithm(name: str,
                       T_i_t: dict[tuple[int, int], nx.DiGraph],
                       src_nodes: list[str],
                       caches: list[str],
                       total_time: int,
                       alpha: float = 1,
                       beta: float = 1, 
                       output: bool = True):
    TSMTA = 0
    bc = 0
    cc = 0
    rc = 0
    if name != "TSMTA":
        bc = BC(T_i_t, src_nodes, total_time)
        if name != "DMTS":
            cc = CC(T_i_t, src_nodes, caches, total_time, alpha)
    else:
        for idx, si in enumerate(src_nodes):
            for t in range(total_time):
                G = T_i_t.get((idx, t))
                if G is None:
                    continue
                for u, v, attrs in G.edges(data=True):
                    TSMTA += attrs.get("cost_traffic", 0.0)
    rc = RC(T_i_t, src_nodes, total_time, beta)
    total = bc + cc + rc + TSMTA

    if output:
        if name != "TSMTA":
            print(f"[{name}] BC={bc:.2f}, CC={cc:.2f}, RC={rc:.2f}, Total={total:.2f}")
        else:
            print(f"[{name}] BC+CC={TSMTA:.2f}, RC={rc:.2f}, Total={total:.2f}")
    return bc, cc, rc, total

def Optimal(T_i_t: dict[tuple[int, int], nx.DiGraph], srcs: list[str], caches: list[str], TIG_Interval: dict[tuple[int, int, int], nx.DiGraph], total_time: int, candidates_amount: int):
    intervals: dict[int, list[tuple[int, int]]] = {}
    G: nx.DiGraph

    for idx, si in enumerate(srcs):
        start = 0
        G = T_i_t[(idx, start)]
        intervals[idx] = []
        for j in range(total_time):
            if not Algorithm.are_graphs_equal(G, T_i_t[(idx, j)]):
                intervals[idx].append((start, j - 1))
                start = j
                G = T_i_t[(idx, start)]
        intervals[idx].append((start, total_time - 1))

    dists: dict
    for idx, si in enumerate(srcs):
        RCL: list = []
        for t1, t2 in intervals[idx]:
            interval_graph = T_i_t[(idx, t1)]
            TIG_t1_t2 = TIG_Interval[(idx, t1, t2)].copy()
            dists_pack, path = Algorithm.dijkstra_min_edges(interval_graph, si, weight=TVM.WEIGHT.value)
            dists = {v: cost for v, (cost, _) in dists_pack.items()}
            dest_nodes = set([n for n, d in interval_graph.nodes(data=True) if d.get("type") == "dest"])
            for d in dest_nodes:
                TIG_t1_t2.remove_edge(path[d][-2], path[d][-1])
            new_dists_pack, new_path = Algorithm.dijkstra_min_edges(TIG_t1_t2, si, weight=TVM.WEIGHT.value)
            new_dists = {v: cost for v, (cost, _) in new_dists_pack.items()}

            for d in dest_nodes:
                RCL.append((new_dists.get(d, INF) - dists[d], path.get(d, None), new_path.get(d, None), (t1, t2)))

        RCL_sorted = sorted(
            RCL,
            key=lambda x: (
                round(x[0], 12),                
                tuple(map(str, x[1] or ())),    
                tuple(map(str, x[2] or ())),    
                x[3]  
            )
        )                                                                                                
        for i in range(min(candidates_amount, len(RCL_sorted))):
            dist, path, new_path, (t1, t2) = RCL_sorted[i]
            if not new_path:
                continue
            cache = {}
            bc, cc, rc, total = evaluate_algorithm("TSMTA", T_i_t, srcs, caches, total_time, beta=0.1, output=False)
            cache[t1] = T_i_t[(idx, t1)].copy()
            new_T_i_t = T_i_t[(idx, t1)].copy()
            new_T_i_t.remove_edge(path[-2], path[-1])
            add_edges = list(zip(new_path[:-1], new_path[1:]))
            for u, v in add_edges:
                if TIG_Interval[(idx, t1, t1)].has_edge(u, v):
                    edge_attr = TIG_Interval[(idx, t1, t1)][u][v]
                    new_T_i_t.add_edge(u, v, **edge_attr)
                else:
                    raise KeyError(
                        f"❌ Edge ({u}->{v}) not found in TIG_Interval[{idx}, {t1}, {t1}]"
                    )
            new_T_i_t = Algorithm.shortest_path_tree(new_T_i_t, si, weight=TVM.WEIGHT.value)
            
            min_val = total
            l_ch, r_ch = -1, 0
            for t_l in range(t1, t2 + 1):
                for t_r in range(t_l, t2 + 1):
                    T_i_t[(idx, t_r)] = new_T_i_t
                    bc, cc, rc, val = evaluate_algorithm("TSMTA", T_i_t, srcs, caches, total_time, beta=0.1, output=False)
                    if val < min_val:
                        min_val = val
                        l_ch = t_l
                        r_ch = t_r
                for t_r in range(t_l, t2 + 1):
                    T_i_t[(idx, t_r)] = cache[t1]

            # print(val, i, new_rc, origin_rc)
            if min_val < total:
                for t in range(l_ch, r_ch + 1):
                    T_i_t[(idx, t)] = new_T_i_t

    # total_time = 5
    # (0, 3) (4, 5) cost = 10, rerouting cost = 5
    # (0, 3) (4, 5) cost = 11 rerouting = 0
    # SPT
    # (0, 3)
    # # a->b->c->d 5
    # a->e->d1 6 weight cost = 6 rerouting cost = 4 cache cost = x
    # a->x->b->d1 weight cost = 6 rerouting cost = 7 cache cost = y
    # a->e->x->c->d1 weight cost = 7 rerouting cost = 0 cache cost = z

    # a->x->e->d2
    # a->e->x->c->e->x->c->d1
    
    # a->y->e->d3
    # a->x->e->x->e->d2

    # (4, 5)
    # a->e->x->c->d1 2 rerouting = 0

    # total cost = 10
    # total cost = 8

    # islooping