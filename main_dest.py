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
        f"results_ns{cfg['n_sats']}"
        f"_nc{cfg['n_clouds']}"
        f"_nd{cfg['start_dests']}"
        f"_p{cfg['num_planes']}"
        f"_t{cfg['total_time']}_avg_std.xlsx" # 檔名更新以反映包含標準差
    )
    
    dir_path = "output_graphs"
    os.makedirs(dir_path, exist_ok=True)
    
    # --- 實驗設定 ---
    NUM_RUNS = 1
    BASE_SEED = 42
    
    algo_names = ["DMTS", "OffPA", "SSSP", "TSMTA"]
    
    # 修改：改為儲存列表 (List)，以便後續計算標準差
    # 結構: algo_results["DMTS"]["BC"] = [run1_val, run2_val, ...]
    algo_results = {
        algo: {"BC": [], "CC": [], "RC": [], "Total": []} 
        for algo in algo_names
    }

    print(f"=== 開始執行 {NUM_RUNS} 次實驗 (將計算平均與變異/標準差) ===")

    for run_idx in range(NUM_RUNS):
        current_seed = BASE_SEED + run_idx
        print(f"\n[Run {run_idx+1}/{NUM_RUNS}] Seed: {current_seed}")
        
        # 1. 設定隨機並生成該次實驗的圖
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

        reachable_dests = [
            d for d in dest_nodes_all
            if nx.has_path(graphs[0], src, d)
        ]
        dest_nodes = set(reachable_dests)

        # 3. 執行演算法並 Append 結果到列表
        
        # --- DMTS ---
        T_DMTS = Execute_DMTS(graphs, time_slots)
        bc, cc, rc, total = TVM.evaluate_algorithm("DMTS", T_DMTS, src_nodes, caches, time_slots)
        algo_results["DMTS"]["BC"].append(bc)
        algo_results["DMTS"]["CC"].append(cc)
        algo_results["DMTS"]["RC"].append(rc)
        algo_results["DMTS"]["Total"].append(total)

        # --- OffPA ---
        T_OffPA = Execute_OffPA(graphs, caches, time_slots)
        bc, cc, rc, total = TVM.evaluate_algorithm("OffPA", T_OffPA, src_nodes, caches, time_slots)
        algo_results["OffPA"]["BC"].append(bc)
        algo_results["OffPA"]["CC"].append(cc)
        algo_results["OffPA"]["RC"].append(rc)
        algo_results["OffPA"]["Total"].append(total)

        # --- SSSP ---
        T_SSSP = Execute_SSSP_Union(graphs, time_slots, src_nodes, dest_nodes)
        bc, cc, rc, total = TVM.evaluate_algorithm("SSSP", T_SSSP, src_nodes, caches, time_slots)
        algo_results["SSSP"]["BC"].append(bc)
        algo_results["SSSP"]["CC"].append(cc)
        algo_results["SSSP"]["RC"].append(rc)
        algo_results["SSSP"]["Total"].append(total)

        # --- TSMTA ---
        T_TSMTA, TIG, TIG_Edges_Map = Execute_TSMTA(graphs, src_nodes, caches, dest_nodes, node_attr_map, time_slots)
        TVM.Optimal(T_TSMTA, src_nodes, caches, TIG, time_slots, 100)
        TVM.expand_virtual_edges(T_i_t=T_TSMTA, TIG_Interval=TIG, TIG_Edges_Map=TIG_Edges_Map, srcs=src_nodes, caches=caches, total_time=time_slots)
        
        bc, cc, rc, total = TVM.evaluate_algorithm("TSMTA", T_TSMTA, src_nodes, caches, time_slots)
        algo_results["TSMTA"]["BC"].append(bc)
        algo_results["TSMTA"]["CC"].append(cc)
        algo_results["TSMTA"]["RC"].append(rc)
        algo_results["TSMTA"]["Total"].append(total)

    # 4. 統計計算與寫入 Excel
    print(f"\n=== 實驗結束，正在寫入 Excel: {excel_path} ===")
    graph_name = f"graph_{cfg['n_dests']}_avg{NUM_RUNS}"

    for algo in algo_names:
        # 取得列表
        vals_bc = algo_results[algo]["BC"]
        vals_cc = algo_results[algo]["CC"]
        vals_rc = algo_results[algo]["RC"]
        vals_total = algo_results[algo]["Total"]

        # 計算 Mean (平均)
        mean_bc = np.mean(vals_bc)
        mean_cc = np.mean(vals_cc)
        mean_rc = np.mean(vals_rc)
        mean_total = np.mean(vals_total)

        # 計算 Std Dev (樣本標準差, ddof=1)
        # 畫 Error bar 時，通常使用標準差
        std_bc = np.std(vals_bc, ddof=1)
        std_cc = np.std(vals_cc, ddof=1)
        std_rc = np.std(vals_rc, ddof=1)
        std_total = np.std(vals_total, ddof=1)

        row = {
            "experiment_id": None,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "graph": graph_name,
            "algo": algo,
            # 平均值
            "BC": mean_bc,
            "CC": mean_cc,
            "RC": mean_rc,
            "Total": mean_total,
            # 標準差 (用於垂直誤差線)
            "BC_Std": std_bc,
            "CC_Std": std_cc,
            "RC_Std": std_rc,
            "Total_Std": std_total
        }
        
        save_result_to_excel(excel_path, row)
        print(f"已儲存 {algo}: Mean={mean_total:.2f}, Std={std_total:.2f}")
        
if __name__ == "__main__":
    main()