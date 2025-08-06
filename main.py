import networkx as nx
from Random_Orbit import print_graphs
from Save_And_Read_Graphs import load_graph_from_txt
import Algorithm

def main():
    graphs = []
    for t in range(10):
        G = load_graph_from_txt(f"graphs/graph_t{t}.txt")
        graphs.append(G)

    results = []
    T = []
    alpha = 0.2
    candidates = 5
    time_slots = 10
    for t, G in enumerate(graphs):
        T.append(Algorithm.LMBBSP_multicast(G, "s0", [node for node, attr in G.nodes(data=True) if attr["type"] == "dest"], alpha=alpha, c=candidates))
    results = Algorithm.DMTS(time_slots=time_slots, graphs=T)
    
    print_graphs(results)
if __name__ == "__main__":
    main()