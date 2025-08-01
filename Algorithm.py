import networkx as nx
from typing import List, Set
import heapq

def MBBSP(graph: nx.DiGraph, s: str, t: str):
    bottleneck = {node: 0 for node in graph.nodes}
    bottleneck[s] = float('inf')
    prev = {}

    # 最大堆（頻寬大的優先）
    heap = [(-bottleneck[s], s)] 

    while heap:
        cur_bw_neg, u = heapq.heappop(heap)
        cur_bw = -cur_bw_neg

        if u == t:
            break

        for v in graph.successors(u):
            edge_bw = graph[u][v]['bandwidth']
            new_bw = min(cur_bw, edge_bw)
            if new_bw > bottleneck[v]:
                bottleneck[v] = new_bw
                prev[v] = u
                heapq.heappush(heap, (-new_bw, v))

    # reconstruct path
    if t not in prev and s != t:
        return None, 0

    path = []
    cur = t
    while cur != s:
        path.append(cur)
        cur = prev[cur]
    path.append(s)
    path.reverse()

    return path, bottleneck[t]


def MBBSP_multicast(graph: nx.DiGraph, s: str, destinations: List[str]):
    tree = nx.DiGraph()
    tree.add_node(s, **graph.nodes[s])
    connected: Set[str] = {s}
    remaining = set(destinations)
    total_min_bottleneck = float('inf')

    while remaining:
        best_path = None
        best_bw = -1
        best_d = None

        # 嘗試從目前已加入的任一節點 u，去連接任一目的地 d
        for u in connected:
            for d in remaining:
                path, bw = MBBSP(graph, u, d)
                if path and bw > best_bw:
                    best_path = path
                    best_bw = bw
                    best_d = d

        if not best_path:
            raise ValueError("No path found to some destination.")

        # 加入這條最好的 path
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            if not tree.has_node(v):
                tree.add_node(v, **graph.nodes[v])
            tree.add_edge(u, v, **graph[u][v])
            connected.add(v)

        remaining.remove(best_d)
        total_min_bottleneck = min(total_min_bottleneck, best_bw)

    return tree, total_min_bottleneck

