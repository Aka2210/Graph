from queue import Queue
import networkx as nx
from typing import List

def MBBSP_multicast(graph: nx.Graph, s: str, destinations: List[str]):
    # Step 1: 建 MaxST
    maxst = nx.maximum_spanning_tree(graph, weight='bandwidth')

    # 和原圖相同型態（Graph / DiGraph）
    multicast_tree = graph.__class__()
    min_bw = float('inf')
    mbb_result = min_bw
    
    # Step 2: 找 source → dest 路徑 & 計算 MBB
    for dest in destinations:
        path = nx.shortest_path(maxst, source=s, target=dest)
        
        for u, v in zip(path[:-1], path[1:]):
            # 複製節點屬性
            if not multicast_tree.has_node(u):
                multicast_tree.add_node(u, **graph.nodes[u])
            if not multicast_tree.has_node(v):
                multicast_tree.add_node(v, **graph.nodes[v])

            # 複製邊屬性（完整）
            if not multicast_tree.has_edge(u, v):
                multicast_tree.add_edge(u, v, **graph[u][v])

            # 更新 MBB
            bw = graph[u][v]['bandwidth']
            min_bw = min(min_bw, bw)

        mbb_result = min_bw

    return mbb_result


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
    