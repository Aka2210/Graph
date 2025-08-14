import networkx as nx

def print_graph(G):
    print("Nodes:")
    for n, attr in G.nodes(data=True):
        x, y, z = attr["pos"]
        print(f"  {n}: ({x:.2f}, {y:.2f}, {z:.2f}) type: {attr["type"]} bw: {attr["bandwidth"]}")
    print("Edges:")
    for u, v, attr in G.edges(data=True):
        print(f"  {u} -> {v}, latency = {attr['latency']:.2f} bw = {attr["bandwidth"]}")

def print_graphs(graphs):
    for t, G in enumerate(graphs):
        print(f"\n=== Time t = {t} ===")
        print_graph(G)
        
def are_graphs_equal(seq1: list[nx.DiGraph], seq2: list[nx.DiGraph]) -> bool:
    if len(seq1) != len(seq2):
        print(f"Length mismatch: {len(seq1)} vs {len(seq2)}")
        return False

    all_match = True  # 用來記錄是否完全一致

    for t, (G_orig, G_loaded) in enumerate(zip(seq1, seq2)):
        try:
            assert set(G_orig.nodes()) == set(G_loaded.nodes()), f"Node mismatch at t={t}"
            assert set(G_orig.edges()) == set(G_loaded.edges()), f"Edge mismatch at t={t}"
        except AssertionError as e:
            print(e)
            all_match = False  # 有不一致，標記為 False

    return all_match