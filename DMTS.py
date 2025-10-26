import networkx as nx
from typing import List
import MBBSP

def LMBBSP_multicast(graph: nx.DiGraph, s: str, destinations: List[str], alpha: float, c: int):
    threshold = max(MBBSP.MBBSP_multicast(graph, s, destinations) * (1 - alpha), graph.nodes[s]["bandwidth"])
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
        last_hops = []
        for dest in destinations:
            cur = dest
            while cur != s and pred[cur]:
                p = pred[cur][0]
                path_edges.add((p, cur))
                if cur == dest:
                    last_hops.append((p, cur))
                cur = p

        subG = H.edge_subgraph(path_edges).copy()
        results.append(subG)

        H.remove_edges_from(last_hops)

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