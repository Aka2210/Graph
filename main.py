from datetime import datetime
import json
import os
import sys
import time
import networkx as nx
import Debug
from Random_Orbit import generate_graph_sequence_realistic
from Save_And_Read_Graphs import load_graph_sequence_from_txt, save_result_to_excel
import Algorithm
import TVM
import PDTA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import DMTS
import OffPA
import random
import numpy as np

DIR_PATH = "output_graphs"
time_slots: int

def Execute_SSSP_Union(graphs: list[nx.Graph], time_slots: int, src_nodes: list[str], dest_nodes: set[str], weight="cost_traffic"):
    s = src_nodes[0]
    results = []
    
    for idx, graph in enumerate(graphs):
        lengths, paths = nx.single_source_dijkstra(graph, source=s, weight=weight)
        T = nx.DiGraph()
        if s in graph:
            T.add_node(s, **dict(graph.nodes[s]))
        else:
            T.add_node(s)
            
        for d in dest_nodes:
            if d not in paths:
                raise nx.NetworkXNoPath(f"[t={idx}] No path from {s} to dest {d}")

            path = paths[d]
            for u, v in zip(path[:-1], path[1:]):
                # node attrs
                if u not in T:
                    T.add_node(u, **dict(graph.nodes[u]))
                if v not in T:
                    T.add_node(v, **dict(graph.nodes[v]))

                edge_attr = dict(graph[u][v]) 

                if not T.has_edge(u, v):
                    T.add_edge(u, v, **edge_attr)
                else:
                    T[u][v].update(edge_attr)
        
        results.append(T)
        
    T_i_t: dict[tuple[int, int], nx.DiGraph] = {}
    for i in range(time_slots):
        T_i_t[(0, i)] = results[i]
    return T_i_t

def Execute_DMTS(graphs: list[nx.Graph], time_slots: int) -> dict[tuple[int, int], nx.DiGraph]:
    start_time = time.time()
    T = []
    alpha = 0.2
    DMTS_candidates = 5
    for t, G in enumerate(graphs):
        src_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "src"]
        dest_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "dest"]
        # src_nodes[0]為單源多目的算法, 直接傳入src_nodes為多源(尚未完成, 需討論)
        T.append(DMTS.LMBBSP_multicast(G, src_nodes[0], dest_nodes, alpha=alpha, c=DMTS_candidates))
    results = DMTS.DMTS(time_slots=time_slots, graphs=T)
    T_i_t: dict[tuple[int, int], nx.DiGraph] = {}
    for i in range(time_slots):
        T_i_t[(0, i)] = results[i]
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f}s")
    return T_i_t

def Execute_OffPA(graphs: list[nx.Graph], caches: list[str], time_slots: int) -> dict[tuple[int, int], nx.DiGraph]:
    start_time = time.time()
    results = OffPA.STARFRONT_sequences(graphs, caches)
    T_i_t: dict[tuple[int, int], nx.DiGraph] = {}
    for i in range(time_slots):
        T_i_t[(0, i)] = results[i]
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f}s")
    return T_i_t

def Execute_TSMTA(graphs: list[nx.Graph], src_nodes: list[str], caches: list[str], dest_nodes: set[str], node_attr_map: list[str], time_slots: int) -> dict[tuple[int, int], nx.DiGraph]:
    start_time = time.time()
    caches = [n for n, d in graphs[0].nodes(data=True) if d.get("cache") == True]
    TIG, CTIG, TIG_Edges_Map, CTIG_Edges_Map = TVM.TIG_CTIG(graphs, src_nodes, caches)
    dests_set = {}
    for idx, si in enumerate(src_nodes):
        for i in range(time_slots):
            for j in range(i, time_slots):
                dests_set[(idx, i, j)] = dest_nodes
    print(time.time() - start_time)
    T_i_t = TVM.TSMTA(TIG, CTIG, TIG_Edges_Map, CTIG_Edges_Map, src_nodes, caches, dests_set, time_slots, node_attr_map)
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f}s")
    return T_i_t, TIG, TIG_Edges_Map
    
def main():
    if len(sys.argv) < 2:
        print("用法: python main.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Excel 檔名
    excel_path = (
        f"results_ns{cfg['n_sats']}"  # 注意: 原本是 start_sats，建議統一用 n_sats
        f"_nc{cfg['n_clouds']}"
        f"_nd{cfg['n_dests']}"
        f"_p{cfg['num_planes']}"
        f"_t{cfg['total_time']}_avg50.xlsx" # 檔名加上 avg50 以示區別
    )
    
    dir_path = "output_graphs"
    os.makedirs(dir_path, exist_ok=True)
    
    # --- 實驗設定 ---
    NUM_RUNS = 50
    BASE_SEED = 42
    
    # 初始化累加器：用來存 4 個演算法的加總結果
    # 結構: algo_sums["DMTS"]["BC"] = 累加值
    algo_names = ["DMTS", "OffPA", "SSSP", "TSMTA"]
    algo_sums = {
        algo: {"BC": 0.0, "CC": 0.0, "RC": 0.0, "Total": 0.0} 
        for algo in algo_names
    }

    print(f"=== 開始執行 {NUM_RUNS} 次實驗並取平均 ===")

    for run_idx in range(NUM_RUNS):
        current_seed = BASE_SEED + run_idx
        print(f"\n[Run {run_idx+1}/{NUM_RUNS}] Seed: {current_seed}")
        
        # 1. 設定隨機並生成該次實驗的圖
        # 建議直接生成 (Generate) 而不是讀檔，確保每次拓樸不同且獨立
        random.seed(current_seed)
        np.random.seed(current_seed)
        os.environ["PYTHONHASHSEED"] = str(current_seed)

        graphs = generate_graph_sequence_realistic(
            seed=current_seed,
            n_sats=cfg["n_sats"],
            n_clouds=cfg["n_clouds"],
            n_srcs=cfg["n_srcs"],
            n_dests=cfg["n_dests"],
            total_time=cfg["total_time"],
            num_planes=cfg["num_planes"],
            altitude_km=cfg["altitude_km"],
            inclination_deg=cfg["inclination_deg"],
            f_phasing_param=cfg["f_phasing_param"],
            base_angular_velocity=cfg["base_angular_velocity"],
            thr_cloud_to_cloud=cfg["thr_cloud_to_cloud"],
            region_dist_thr=cfg["region_dist_thr"]
        )
        
        # 2. 提取該次圖形的節點資訊
        src_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "src"]
        dest_nodes_all = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "dest"]
        caches = [n for n, d in graphs[0].nodes(data=True) if d.get("cache") == True]
        node_attr_map = {n: dict(d) for n, d in graphs[0].nodes(data=True)}
        time_slots = len(graphs)
        src = src_nodes[0]

        # 過濾不可達的目的地 (避免算法報錯)
        reachable_dests = [
            d for d in dest_nodes_all
            if nx.has_path(graphs[0], src, d)
        ]
        dest_nodes = set(reachable_dests)

        # 3. 執行各個演算法並取得該次 Cost
        
        # --- DMTS ---
        T_DMTS = Execute_DMTS(graphs, time_slots)
        bc, cc, rc, total = TVM.evaluate_algorithm("DMTS", T_DMTS, src_nodes, caches, time_slots)
        algo_sums["DMTS"]["BC"] += bc
        algo_sums["DMTS"]["CC"] += cc
        algo_sums["DMTS"]["RC"] += rc
        algo_sums["DMTS"]["Total"] += total

        # --- OffPA ---
        T_OffPA = Execute_OffPA(graphs, caches, time_slots)
        bc, cc, rc, total = TVM.evaluate_algorithm("OffPA", T_OffPA, src_nodes, caches, time_slots)
        algo_sums["OffPA"]["BC"] += bc
        algo_sums["OffPA"]["CC"] += cc
        algo_sums["OffPA"]["RC"] += rc
        algo_sums["OffPA"]["Total"] += total

        # --- SSSP ---
        T_SSSP = Execute_SSSP_Union(graphs, time_slots, src_nodes, dest_nodes)
        bc, cc, rc, total = TVM.evaluate_algorithm("SSSP", T_SSSP, src_nodes, caches, time_slots)
        algo_sums["SSSP"]["BC"] += bc
        algo_sums["SSSP"]["CC"] += cc
        algo_sums["SSSP"]["RC"] += rc
        algo_sums["SSSP"]["Total"] += total

        # --- TSMTA ---
        T_TSMTA, TIG, TIG_Edges_Map = Execute_TSMTA(graphs, src_nodes, caches, dest_nodes, node_attr_map, time_slots)
        # 執行 TSMTA 的優化步驟
        TVM.Optimal(T_TSMTA, src_nodes, caches, TIG, time_slots, 100)
        TVM.expand_virtual_edges(T_i_t=T_TSMTA, TIG_Interval=TIG, TIG_Edges_Map=TIG_Edges_Map, srcs=src_nodes, caches=caches, total_time=time_slots)
        
        bc, cc, rc, total = TVM.evaluate_algorithm("TSMTA", T_TSMTA, src_nodes, caches, time_slots)
        algo_sums["TSMTA"]["BC"] += bc
        algo_sums["TSMTA"]["CC"] += cc
        algo_sums["TSMTA"]["RC"] += rc
        algo_sums["TSMTA"]["Total"] += total

    # 4. 迴圈結束後，計算平均並存檔
    print(f"\n=== 實驗結束，正在寫入 Excel: {excel_path} ===")
    graph_name = f"graph_{cfg['n_sats']}_avg{NUM_RUNS}"

    for algo in algo_names:
        # 計算平均
        avg_bc = algo_sums[algo]["BC"] / NUM_RUNS
        avg_cc = algo_sums[algo]["CC"] / NUM_RUNS
        avg_rc = algo_sums[algo]["RC"] / NUM_RUNS
        avg_total = algo_sums[algo]["Total"] / NUM_RUNS

        row = {
            "experiment_id": None,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "graph": graph_name,
            "algo": algo,
            "BC": avg_bc,
            "CC": avg_cc,
            "RC": avg_rc,
            "Total": avg_total
        }
        
        save_result_to_excel(excel_path, row)
        print(f"已儲存 {algo}: Total={avg_total:.2f}")
        
if __name__ == "__main__":
    main()