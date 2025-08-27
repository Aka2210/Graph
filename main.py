import os
import networkx as nx
from Debug import print_graphs
from Debug import print_graph
from Save_And_Read_Graphs import load_graph_sequence_from_txt
import Algorithm

DIR_PATH = "output_graphs"

def main():
    dir_path = "output_graphs"
    os.makedirs(dir_path, exist_ok=True)

    # 取得資料夾內的 txt 檔數量
    txt_count = len([f for f in os.listdir(dir_path) if f.endswith(".txt")])
    graphs = load_graph_sequence_from_txt(path=DIR_PATH, idx=txt_count)

    results = []
    T = []
    alpha = 0.2
    DMTS_candidates = 5
    time_slots = len(graphs)
    # for t, G in enumerate(graphs):
    #     src_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "src"]
    #     dest_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "dest"]
    #     # src_nodes[0]為單源多目的算法, 直接傳入src_nodes為多源(尚未完成, 需討論)
    #     T.append(Algorithm.LMBBSP_multicast(G, src_nodes[0], dest_nodes, alpha=alpha, c=DMTS_candidates))
    # results = Algorithm.DMTS(time_slots=time_slots, graphs=T)
    # print_graphs(results)
    
    # print_graphs(Algorithm.STARFRONT_sequences(graphs))
    src_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "src"]
    dest_nodes = set([n for n, d in graphs[0].nodes(data=True) if d.get("type") == "dest"])
    satellites_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "satellite"]
    cloud_nodes = [n for n, d in graphs[0].nodes(data=True) if d.get("type") == "cloud"]
    TIG, CTIG = Algorithm.TIG_CTIG(graphs, src_nodes, satellites_nodes + cloud_nodes)
    dests_set = {}
    for idx, si in enumerate(src_nodes):
        for i in range(time_slots):
            for j in range(i, time_slots):
                dests_set[(idx, i, j)] = dest_nodes
    Algorithm.TSMTA(TIG, CTIG, None, None, src_nodes, dests_set, len(graphs))
    # print_graph(Algorithm.PDTA(2, src_nodes[0], len(dest_nodes), dest_nodes, graphs[0]))
if __name__ == "__main__":
    main()