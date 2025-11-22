from datetime import datetime
import json
import os
import sys
import time
import networkx as nx
import Debug
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

    # ⭐ 自動根據 config 產生 Excel 名稱
    excel_path = (
        f"results_ns{cfg['n_sats']}"
        f"_nc{cfg['n_clouds']}"
        f"_nd{cfg['n_dests']}"
        f"_p{cfg['num_planes']}"
        f"_t{cfg['total_time']}.xlsx"
    )
    
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)
    
    dir_path = "output_graphs"
    os.makedirs(dir_path, exist_ok=True)

    # 取得資料夾內的 txt 檔數量
    txt_count = len([f for f in os.listdir(dir_path) if f.endswith(".txt")])
    graphs = load_graph_sequence_from_txt(path=DIR_PATH, idx=txt_count)
    
    src_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "src"]
    dest_nodes = set([n for n, d in graphs[0].nodes(data=True) if d.get("type") == "dest"])
    satellites_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "satellite"]
    cloud_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "cloud"]
    caches = [n for n, d in graphs[0].nodes(data=True) if d.get("cache") == True]
    node_attr_map = {n: dict(d) for n, d in graphs[0].nodes(data=True)}
    time_slots = len(graphs)
    
    src = src_nodes[0]
    reachable_dests = [
        d for d in dest_nodes
        if nx.has_path(graphs[0], src, d)  # 保留方向性（DiGraph）
    ]
    if len(reachable_dests) < len(dest_nodes):
        print(f"[警告] 僅 {len(reachable_dests)} / {len(dest_nodes)} 個目的地可從 {src} 抵達。")
        print(f"不可達目的地: {set(dest_nodes) - set(reachable_dests)}")

    dest_nodes = set(reachable_dests)
    # for G in graphs:
    #     Debug.draw_graph_2d(G, src_nodes[0], 0)
    # DMTS
    T_DMTS = Execute_DMTS(graphs, time_slots)
    # TVM.evaluate_algorithm("DMTS", T_DMTS, src_nodes, caches, time_slots)

    # OffPA
    T_OffPA = Execute_OffPA(graphs, caches, time_slots)
    # TVM.evaluate_algorithm("OffPA", T_OffPA, src_nodes, caches, time_slots)

    # TSMTA
    T_TSMTA, TIG, TIG_Edges_Map = Execute_TSMTA(graphs, src_nodes, caches, dest_nodes, node_attr_map, time_slots)
    # for i in range(time_slots):
    #     for j in range(i, time_slots):
    #         Debug.draw_graph_2d(TIG[(0, i, j)], src_nodes[0], i)
    # TVM.evaluate_algorithm("TSMTA", T_TSMTA, src_nodes, caches, time_slots, beta=0.1)
    # for i in range(time_slots):
    #     Debug.draw_graph_2d(T_TSMTA[(0, i)], src_nodes[0], i)   

    TVM.Optimal(T_TSMTA, src_nodes, caches, TIG, time_slots, 100)
    # for i in range(time_slots):
    #     Debug.draw_graph_2d(T_TSMTA[(0, i)], src_nodes[0], i)
    # TVM.evaluate_algorithm("TSMTA", T_TSMTA, src_nodes, caches, time_slots, beta=0.1)
    TVM.expand_virtual_edges(T_i_t=T_TSMTA, TIG_Interval=TIG, TIG_Edges_Map=TIG_Edges_Map, srcs=src_nodes, caches=caches, total_time=time_slots)
    
    graph_name = f"graph_{txt_count}"

    algo_results = {
        "DMTS": T_DMTS,
        "OffPA": T_OffPA,
        "TSMTA": T_TSMTA
    }

    for algo, T_i_t in algo_results.items():

        bc, cc, rc, total = TVM.evaluate_algorithm(
            algo, T_i_t, src_nodes, caches, time_slots
        )

        # 一筆要 append 的 Row
        row = {
            "experiment_id": None,  # save_result_to_excel() 會自動補
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "graph": graph_name,
            "algo": algo,
            "BC": bc,
            "CC": cc,
            "RC": rc,
            "Total": total
        }

        save_result_to_excel(excel_path, row)
if __name__ == "__main__":
    main()