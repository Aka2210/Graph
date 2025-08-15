import networkx as nx

def print_graph(G):
    print("Nodes:")
    for n, attr in G.nodes(data=True):
        print(f"  {n}:")
        for k, v in attr.items():
            print(f"    {k} = {v}")
    print("Edges:")
    for u, v, attr in G.edges(data=True):
        print(f"  {u} -> {v}:")
        for k, v in attr.items():
            print(f"    {k} = {v}")

def print_graphs(graphs):
    for t, G in enumerate(graphs):
        print(f"\n=== Time t = {t} ===")
        print_graph(G)

def are_graphs_equal(seq1: list[nx.DiGraph], seq2: list[nx.DiGraph]) -> bool:
    if len(seq1) != len(seq2):
        print(f"Length mismatch: {len(seq1)} vs {len(seq2)}")
        return False

    all_match = True
    for t, (G1, G2) in enumerate(zip(seq1, seq2)):
        # 比對節點
        if set(G1.nodes) != set(G2.nodes):
            print(f"Node set mismatch at t={t}")
            all_match = False
        else:
            for n in G1.nodes:
                if G1.nodes[n] != G2.nodes[n]:
                    print(f"Node attr mismatch at t={t}, node={n}")
                    print("  G1:", G1.nodes[n])
                    print("  G2:", G2.nodes[n])
                    all_match = False

        # 比對邊
        if set(G1.edges) != set(G2.edges):
            print(f"Edge set mismatch at t={t}")
            all_match = False
        else:
            for e in G1.edges:
                if G1.edges[e] != G2.edges[e]:
                    print(f"Edge attr mismatch at t={t}, edge={e}")
                    print("  G1:", G1.edges[e])
                    print("  G2:", G2.edges[e])
                    all_match = False

    return all_match
