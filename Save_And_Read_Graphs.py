import networkx as nx

def save_graph_to_txt(G, filename):
    with open(filename, "w") as f:
        # === Nodes ===
        f.write("# Nodes\n")
        for n, attr in G.nodes(data=True):
            x, y, z = attr["pos"]      
            f.write(f"{n} {attr["type"]} {x:.6f} {y:.6f} {z:.6f} {attr['bandwidth']}\n")

        # === Edges ===
        f.write("# Edges\n")
        for u, v, attr in G.edges(data=True):
            weight = attr["weight"]
            bandwidth = attr["bandwidth"]
            f.write(f"{u} {v} {weight:.6f} {bandwidth}\n")
            
def load_graph_from_txt(filename: str) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(filename, "r") as f:
        lines = f.readlines()
    
    mode = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if "# Nodes" in line:
                mode = "node"
            elif "# Edges" in line:
                mode = "edge"
            continue
        
        if mode == "node":
            name, typ, x, y, z, bw = line.split()
            G.add_node(name, pos=(float(x), float(y), float(z)), type=typ, bandwidth=int(bw))
        elif mode == "edge":
            u, v, weight, bandwidth = line.split()
            G.add_edge(u, v, weight=float(weight), bandwidth=int(bandwidth))
    
    return G