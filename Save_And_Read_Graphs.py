import os
import networkx as nx

def save_graph_sequence_to_txt(graph_seq, dir_path="output_graphs"):
    """
    將一串時間序列圖 graph_seq 存成單一 txt：
    檔名：graph_testbed{dir_path 內現有檔數+1}.txt
    格式：
    # Time t
    # Nodes
    name type x y z bandwidth time
    # Edges
    u v latency edge_bw time
    """
    os.makedirs(dir_path, exist_ok=True)
    count = sum(1 for f in os.listdir(dir_path)
                if f.startswith("graph_testbed") and f.endswith(".txt"))
    filename = os.path.join(dir_path, f"graph_testbed{count+1}.txt")

    with open(filename, "w") as f:
        for G in graph_seq:
            t = G.graph.get("time", None)
            if t is None:
                raise ValueError("Graph has no graph-level time and no node time info.")

            f.write(f"# Time {t}\n")

            # === Nodes ===
            f.write("# Nodes\n")
            for n, attr in G.nodes(data=True):
                pos = attr.get("pos", (0.0, 0.0, 0.0))
                x, y, z = pos
                typ = attr.get("type", "unk")
                bw  = int(attr.get("bandwidth", 0))
                f.write(f"{n} {typ} {x:.6f} {y:.6f} {z:.6f} {bw}\n")

            # === Edges ===
            f.write("# Edges\n")
            for u, v, attr in G.edges(data=True):
                latency = float(attr.get("latency", attr.get("weight", 0.0)))
                ebw     = int(attr.get("bandwidth", 0))
                f.write(f"{u} {v} {latency:.6f} {ebw}\n")

    return filename

def load_graph_sequence_from_txt(path: str, idx: int | None = None) -> list[nx.DiGraph]:
    """
    讀回由 save_graph_sequence_to_txt() 產生的檔案，
    回傳 List[DiGraph]，每個 DiGraph 代表一個 time slot。
    """
    if idx is None:
        filename = path
    else:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"期望資料夾但拿到：{path}")
        filename = os.path.join(path, f"graph_testbed{idx}.txt")

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"找不到檔案：{filename}")
    
    graphs: list[nx.DiGraph] = []
    G = None
    mode = None
    current_time = None
    
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                if line.startswith("# Time"):
                    # 遇到新時間段，先把上一個 G 收起來
                    if G is not None:
                        graphs.append(G)
                    # 解析 time
                    try:
                        current_time = int(line.split()[-1])
                    except Exception:
                        current_time = -1
                    G = nx.DiGraph(time=current_time)
                    mode = None
                elif line == "# Nodes":
                    mode = "node"
                elif line == "# Edges":
                    mode = "edge"
                continue

            if mode == "node":
                # name typ x y z bw time
                name, typ, x, y, z, bw = line.split()
                G.add_node(
                    name,
                    pos=(float(x), float(y), float(z)),
                    type=typ,
                    bandwidth=int(bw)
                )
            elif mode == "edge":
                # u v latency edge_bw time
                u, v, latency, ebw = line.split()
                G.add_edge(u, v,
                    latency=float(latency),
                    bandwidth=int(ebw)
                )

    # 收最後一個圖
    if G is not None:
        graphs.append(G)
    return graphs