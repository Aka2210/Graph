from asyncio import Queue
from collections import defaultdict, deque
from enum import Enum
import math
import random
import sys
import time
import networkx as nx
import heapq
from typing import Any, Dict, List, Set, Tuple
import Debug
import Algorithm

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

        # 先過濾掉 bottleneck 為 0（不可達）的節點
        reachable_dests = [d for d in remaining if bottleneck[d] > 0]
        if not reachable_dests:
            sys.exit("[警告] MBBSP_multicast: 所有剩餘目的地皆不可達，提早結束。")

        best_d = max(reachable_dests, key=lambda d: bottleneck[d])
        best_bw = bottleneck[best_d]

        remaining.remove(best_d)
        total_min_bottleneck = min(total_min_bottleneck, best_bw)

    # 若完全沒有任何可達目的地
    if total_min_bottleneck == float('inf'):
        raise ValueError("No path found to any destination.")

    return total_min_bottleneck
