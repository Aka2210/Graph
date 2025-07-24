import networkx as nx
import matplotlib.pyplot as plt
import random


def build_random_digraph(num_nodes=50, num_edges=125, wmin=1, wmax=20, seed=None):
    """建立隨機有向圖，邊權儲存在 'weight' 屬性。"""
    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    while G.number_of_edges() < num_edges:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u != v and not G.has_edge(u, v):
            w = rng.randint(wmin, wmax)
            G.add_edge(u, v, weight=w)
    return G

def export_edges_array_style(G, path, attr='weight', default=1):
    """
    將有向圖 G 的邊輸出成:
    [
    [u,v,w],
    [u2,v2,w2],
    ...
    ]
    attr: 邊屬性名稱 (預設 'weight')
    default: 若該屬性不存在時使用的成本
    """
    edges = []
    for u, v, data in G.edges(data=True):
        w = data.get(attr, default)
        edges.append((u, v, w))

    with open(path, "w", encoding="utf-8") as f:
        for i, (u, v, w) in enumerate(edges):
            f.write(f"{u} {v} {w}\n")

def draw_graph(G, layout_seed=42, title="Random Directed Graph"):
    """畫出圖，顯示邊權。"""
    pos = nx.spring_layout(G, seed=layout_seed)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.show()


def main():
    # 參數可依需求調整
    num_nodes = 5
    num_edges = 10
    seed = None          # 設成整數即可重現；None 表每次不同

    G = build_random_digraph(num_nodes=num_nodes,
                             num_edges=num_edges,
                             seed=seed)

    for u, v, attrs in G.edges(data=True):
        print(u, "->", v, "cost =", attrs.get('weight'))

    export_edges_array_style(G, "edges.txt", attr='weight')
    # draw_graph(G, layout_seed=42)

if __name__ == "__main__":
    main()
