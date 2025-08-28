from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from typing import Any, List, Set
import Algorithm

INF = float("inf")

class OffPA(Enum):
    USER = "dest"
    SAT = "satellite"
    CLOUD = "cloud"
    SRC = "src"
    REGION = "region"
    WEIGHT = "latency"

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
                res += Algorithm.cost_cache(attr, W_total)
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
            res = Algorithm.multi_src_to_one_dest_subgraph(
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
            extend = Algorithm.multi_src_to_one_dest_subgraph(G, (list(candidate[j].keys()) + srcs), j, OffPA.WEIGHT.value, True)
            tmp_DG = Algorithm.add_nodes_edges_with_attrs(tmp_DG, G, 
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