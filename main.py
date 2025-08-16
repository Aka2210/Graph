import networkx as nx
from Debug import print_graphs
from Save_And_Read_Graphs import load_graph_sequence_from_txt
import Algorithm

DIR_PATH = "output_graphs"

def main():
    graphs = load_graph_sequence_from_txt(path=DIR_PATH, idx=1)

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
    
    print_graphs(Algorithm.STARFRONT_sequences(graphs))
if __name__ == "__main__":
    main()