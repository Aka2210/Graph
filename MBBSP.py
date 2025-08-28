from asyncio import Queue
from collections import defaultdict, deque
from enum import Enum
import math
import random
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

        best_d = max(remaining, key=lambda d: bottleneck[d])
        best_bw = bottleneck[best_d]

        if best_bw == 0:
            raise ValueError("No path found to some destination.")

        remaining.remove(best_d)
        total_min_bottleneck = min(total_min_bottleneck, best_bw)

    return total_min_bottleneck