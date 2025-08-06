import networkx as nx
import heapq
from Random_Orbit import print_graph
from typing import List, Set, Tuple

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

def DMTS(time_slots: int, graphs: List[List[nx.Graph]]):
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