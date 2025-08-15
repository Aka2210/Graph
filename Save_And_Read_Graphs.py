import os
import networkx as nx

def save_graph_sequence_to_txt(graph_seq, dir_path="output_graphs"):
    """
    儲存一串時間序列的 DiGraph
    - 每個 time slot 存成一段
    - Node / Edge 的所有 attr 都會完整保存
    - 存檔格式是 Python dict，可直接 eval 還原
    """
    os.makedirs(dir_path, exist_ok=True)
    count = sum(1 for f in os.listdir(dir_path)
                if f.startswith("graph_testbed") and f.endswith(".txt"))
    filename = os.path.join(dir_path, f"graph_testbed{count+1}.txt")

    with open(filename, "w") as f:
        for G in graph_seq:
            t = G.graph.get("time", None)
            if t is None:
                raise ValueError("Graph has no graph-level time")

            f.write(f"# Time {t}\n")

            # === Nodes ===
            f.write("# Nodes\n")
            for n, attr in G.nodes(data=True):
                f.write(f"{n} {repr(dict(attr))}\n")

            # === Edges ===
            f.write("# Edges\n")
            for u, v, attr in G.edges(data=True):
                f.write(f"{u} {v} {repr(dict(attr))}\n")

    return filename


def load_graph_sequence_from_txt(path: str, idx: int | None = None) -> list[nx.DiGraph]:
    """
    從 save_graph_sequence_to_txt 存的檔案讀取回 List[DiGraph]
    """
    if idx is None:
        filename = path
    else:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"期望資料夾但拿到：{path}")
        filename = os.path.join(path, f"graph_testbed{idx}.txt")

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"找不到檔案：{filename}")
    
    graphs = []
    G = None
    mode = None

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                if line.startswith("# Time"):
                    # 收前一張圖
                    if G is not None:
                        graphs.append(G)
                    try:
                        t = int(line.split()[-1])
                    except:
                        t = -1
                    G = nx.DiGraph(time=t)
                    mode = None
                elif line == "# Nodes":
                    mode = "node"
                elif line == "# Edges":
                    mode = "edge"
                continue

            if mode == "node":
                name, attr_repr = line.split(" ", 1)
                attrs = eval(attr_repr)  # 直接還原成 dict
                G.add_node(name, **attrs)

            elif mode == "edge":
                u, v, attr_repr = line.split(" ", 2)
                attrs = eval(attr_repr)  # 直接還原成 dict
                G.add_edge(u, v, **attrs)

    # 收最後一張圖
    if G is not None:
        graphs.append(G)

    return graphs
