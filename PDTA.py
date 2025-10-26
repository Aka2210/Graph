import networkx as nx
from typing import Set
import Debug
import Algorithm

INF = float("inf")
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

def PDTA_Origin(level: int, r: str, m: int, terminals: Set[str], G: nx.DiGraph) -> nx.DiGraph:
    T_return = nx.DiGraph()
    T_terminals = set(terminals)

    if m <= 0 or not T_terminals or level < 1:
        return T_return

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
        return T_return
        
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
                tmp = PDTA_Origin(level-1, v, n, T_terminals, G)
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
            return T_return
        D_current |= D_min
        T_terminals -= D_min
        T_return = Algorithm.union_graphs(T_return, T_min)

    return T_return

def PDTA(level: int, r: str, m: int, terminals: Set[str], G: nx.DiGraph):
    T_return = nx.DiGraph()
    T_terminals = set(terminals)
    d_T_min_return = INF
    T_record = {}

    dist = dict(nx.single_source_shortest_path(G, r, cutoff=level))
    reachable = set(dist.keys())
    T_terminals = T_terminals & reachable

    if m <= 0 or not T_terminals or level < 1:
        return T_return, d_T_min_return, T_record

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
        return T_return, d_T_min_return, T_record
        
    D_current = set()
    while len(D_current) < m and T_terminals:
        # print(len(D_current), m, T_terminals)
        T_min = nx.DiGraph()
        d_T_min = INF
        tmp: nx.DiGraph
        D_min = set()

        for v in G.successors(r):
            tmp, density, records = PDTA(level-1, v, min(len(T_terminals), m), T_terminals, G)
            tmp.add_node(r, **G.nodes[r])
            tmp.add_node(v, **G.nodes[v])
            tmp.add_edge(r, v, **G[r][v])
            
            d_tmp = PDTA_Density(tmp, 1, T_terminals) 
            if d_tmp < d_T_min:
                T_min = tmp
                d_T_min = d_tmp
            sorted_records = sorted(records.items(), key=lambda x: x[0])
            total_dests = sum(key[1] for key, _ in sorted_records)
            tmp = nx.DiGraph()
            cnt = 0
            idx = 0
            for n in range(1, min(len(T_terminals), m)):
                if n > total_dests:
                    break
                if cnt >= n:
                    continue
            
                while cnt < n:
                    key, G_sub = sorted_records[idx]
                    tmp = Algorithm.union_graphs(tmp, G_sub)
                    cnt += key[1]
                    idx += 1

                d_tmp = PDTA_Density(tmp, 1, T_terminals) 
                if d_tmp < d_T_min:
                    T_min = tmp
                    d_T_min = d_tmp
        D_min = {n for n in T_min.nodes if G.nodes[n].get("type") == "dest"}
        D_min = D_min & T_terminals
        D_current = D_current | D_min
        T_terminals = T_terminals - D_min
        T_return = Algorithm.union_graphs(T_return, T_min)
        d_T_min_return = d_T_min
        T_record[(d_T_min, len(D_min))] = T_min

    return T_return, d_T_min, T_record