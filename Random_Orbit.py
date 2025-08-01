import argparse
import random
import numpy as np
import networkx as nx
from Save_And_Read_Graphs import save_graph_to_txt

def spherical_to_cartesian(radius, inclination, azimuth):
    x = radius * np.sin(inclination) * np.cos(azimuth)
    y = radius * np.sin(inclination) * np.sin(azimuth)
    z = radius * np.cos(inclination)
    return x, y, z

def add_ring_edges(G, prefix, n):
    for i in range(n):
        u = f"{prefix}{i}"
        v = f"{prefix}{(i + 1) % n}"
        pos_u = G.nodes[u]["pos"]
        pos_v = G.nodes[v]["pos"]
        dist = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        G.add_edge(u, v, weight=dist, bandwidth=random.randint(0, int(2 * dist)))
        G.add_edge(v, u, weight=dist, bandwidth=random.randint(0, int(2 * dist)))
        
def add_edges(G, s, d, n, m):
    for ai in range(n):
            for bi in range(m):
                s_name = f"{s}{ai}"
                d_name = f"{d}{bi}"
                dist = np.linalg.norm(np.array(G.nodes[s_name]["pos"]) - np.array(G.nodes[d_name]["pos"]))
                G.add_edge(s_name, d_name, weight=dist, bandwidth=random.randint(0, int(2 * dist)))

def generate_restricted_graph_sequence(
    num_satellites_per_orbit=5,
    num_sources=5,
    num_destinations=5,
    radius=5,
    inclination_deg_1=45,
    inclination_deg_2=135,
    angular_velocity=0.05,
    total_time=10,
    sd_radius=5
):
    inclination1 = np.deg2rad(inclination_deg_1)
    inclination2 = np.deg2rad(inclination_deg_2)
    delta_phi = 2 * np.pi / num_satellites_per_orbit
    initial_phis = [i * delta_phi for i in range(num_satellites_per_orbit)]

    source_positions = [
        (sd_radius * np.cos(2 * np.pi * i / num_sources),
         sd_radius * np.sin(2 * np.pi * i / num_sources),
         0)
        for i in range(num_sources)
    ]
    dest_positions = [
        (sd_radius * np.cos(2 * np.pi * i / num_destinations + np.pi / num_destinations),
         sd_radius * np.sin(2 * np.pi * i / num_destinations + np.pi / num_destinations),
         0)
        for i in range(num_destinations)
    ]

    graph_sequence = []

    for t in range(total_time):
        G = nx.DiGraph()

        # === Add relay A ===
        for i, phi0 in enumerate(initial_phis):
            phi_t = phi0 + angular_velocity * t
            x, y, z = spherical_to_cartesian(radius, inclination1, phi_t)
            name = f"A{i}"
            G.add_node(name, pos=(x, y, z), time=t, type="satellite")

        # === Add cache B ===
        for i, phi0 in enumerate(initial_phis):
            phi_t = phi0 + angular_velocity * t
            x, y, z = spherical_to_cartesian(radius, inclination2, phi_t)
            name = f"B{i}"
            G.add_node(name, pos=(x, y, z), time=t, type="cache")

        # === Add sources ===
        for i, pos in enumerate(source_positions):
            name = f"s{i}"
            G.add_node(name, pos=pos, time=t, type="src")

        # === Add destinations ===
        for i, pos in enumerate(dest_positions):
            name = f"d{i}"
            G.add_node(name, pos=pos, time=t, type="dest")

        # === s -> A ===
        add_edges(G, "s", "A", num_sources, num_satellites_per_orbit)
        
        # === A -> B ===
        add_edges(G, "A", "B", num_satellites_per_orbit, num_satellites_per_orbit)
        
        # === B -> d ===
        add_edges(G, "B", "d", num_satellites_per_orbit, num_destinations)
        
        # === A[i] <-> A[(i+1)%n] === 環狀單向連線
        add_ring_edges(G, "A", num_satellites_per_orbit)
        
        # === B[i] <-> B[(i+1)%n] === 環狀單向連線
        add_ring_edges(G, "B", num_satellites_per_orbit)

        graph_sequence.append(G)

    return graph_sequence

def print_graph(graphs, save=False):
    for t, G in enumerate(graphs):
        print(f"\n=== Time t = {t} ===")
        print("Nodes:")
        for n, attr in G.nodes(data=True):
            x, y, z = attr["pos"]
            print(f"  {n}: ({x:.2f}, {y:.2f}, {z:.2f}) type: {attr["type"]}")
        print("Edges:")
        for u, v, attr in G.edges(data=True):
            print(f"  {u} -> {v}, dist = {attr['weight']:.2f} bw = {attr["bandwidth"]}")
        
        if(save):
            save_graph_to_txt(G, f"graphs/graph_t{t}.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', action='store_true', help='儲存生成的Graphs')
    args = parser.parse_args()
    
    graphs = generate_restricted_graph_sequence(
        num_satellites_per_orbit=5,
        num_sources=1,
        num_destinations=5,
        total_time=10
    )

    print_graph(graphs, args.store)

if __name__ == "__main__":
    main()
