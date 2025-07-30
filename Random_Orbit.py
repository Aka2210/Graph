import numpy as np
import networkx as nx
from itertools import combinations

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
        G.add_edge(u, v, weight=dist)
        G.add_edge(v, u, weight=dist)

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
            G.add_node(name, pos=(x, y, z), time=t)

        # === Add cache B ===
        for i, phi0 in enumerate(initial_phis):
            phi_t = phi0 + angular_velocity * t
            x, y, z = spherical_to_cartesian(radius, inclination2, phi_t)
            name = f"B{i}"
            G.add_node(name, pos=(x, y, z), time=t)

        # === Add sources ===
        for i, pos in enumerate(source_positions):
            name = f"s{i}"
            G.add_node(name, pos=pos, time=t)

        # === Add destinations ===
        for i, pos in enumerate(dest_positions):
            name = f"d{i}"
            G.add_node(name, pos=pos, time=t)

        # === s -> A ===
        for si in range(num_sources):
            s_name = f"s{si}"
            for ai in range(num_satellites_per_orbit):
                a_name = f"A{ai}"
                dist = np.linalg.norm(np.array(G.nodes[s_name]["pos"]) - np.array(G.nodes[a_name]["pos"]))
                G.add_edge(s_name, a_name, weight=dist)

        n = num_satellites_per_orbit
        
        # === A[i] <-> A[(i+1)%n] === 環狀單向連線
        add_ring_edges(G, "A", n)
        
        # === B[i] <-> B[(i+1)%n] === 環狀單向連線
        add_ring_edges(G, "B", n)
        
        # === A -> B ===
        for ai in range(num_satellites_per_orbit):
            for bi in range(num_satellites_per_orbit):
                a_name = f"A{ai}"
                b_name = f"B{bi}"
                dist = np.linalg.norm(np.array(G.nodes[a_name]["pos"]) - np.array(G.nodes[b_name]["pos"]))
                G.add_edge(a_name, b_name, weight=dist)

        # === B -> d ===
        for bi in range(num_satellites_per_orbit):
            b_name = f"B{bi}"
            for di in range(num_destinations):
                d_name = f"d{di}"
                dist = np.linalg.norm(np.array(G.nodes[b_name]["pos"]) - np.array(G.nodes[d_name]["pos"]))
                G.add_edge(b_name, d_name, weight=dist)

        graph_sequence.append(G)

    return graph_sequence

def main():
    graphs = generate_restricted_graph_sequence(
        num_satellites_per_orbit=5,
        num_sources=5,
        num_destinations=5,
        total_time=1  # 減少輸出量觀察
    )

    for t, G in enumerate(graphs):
        print(f"\n=== Time t = {t} ===")
        print("Nodes:")
        for n, attr in G.nodes(data=True):
            x, y, z = attr["pos"]
            print(f"  {n}: ({x:.2f}, {y:.2f}, {z:.2f})")
        print("Edges:")
        for u, v, attr in G.edges(data=True):
            print(f"  {u} -> {v}, dist = {attr['weight']:.2f}")

if __name__ == "__main__":
    main()
