import argparse
import math
import os
import random
import numpy as np
import networkx as nx
from Save_And_Read_Graphs import save_graph_sequence_to_txt
from Save_And_Read_Graphs import load_graph_sequence_from_txt
from Debug import print_graphs
from Debug import are_graphs_equal

def spherical_to_cartesian(radius, inclination, azimuth):
    x = radius * np.sin(inclination) * np.cos(azimuth)
    y = radius * np.sin(inclination) * np.sin(azimuth)
    z = radius * np.cos(inclination)
    return x, y, z
                
def spherical_to_cartesian(r, inc_rad, phi_rad):
    # 極座標：半徑 r、傾角 inc（相當於軌道傾角）、軌道相位 phi
    # 回傳 3D 直角座標
    x = r * math.cos(phi_rad) * math.cos(inc_rad)
    y = r * math.sin(phi_rad) * math.cos(inc_rad)
    z = r * math.sin(inc_rad)
    return x, y, z

def euclid_latency(p1, p2):
    return float(math.dist(p1, p2))

def realistic_latency(p1, p2, u_type, v_type):
    """
    計算合理的鏈路延遲 (ms)
    - p1, p2: 節點座標 (x,y,z)，單位假設為 km
    - u_type, v_type: 節點類型 (src, dest, satellite, cloud)
    """
    # 距離 (km -> m)
    distance_km = float(math.dist(p1, p2))
    d = distance_km * 1000.0

    # 傳播速度
    if u_type == "cloud" and v_type == "cloud":
        speed = 2e8  # 光纖，大約 2/3 c
    else:
        speed = 3e8  # 空間鏈路，接近真空光速

    # 傳播延遲 (ms)
    prop_delay = d / speed * 1000.0

    # 處理延遲 (ms)
    if u_type == "satellite" and v_type == "satellite":
        proc_delay = 1.0    # ISL 轉發延遲
    elif "cloud" in (u_type, v_type):
        proc_delay = 0.5    # 地面節點處理
    else:
        proc_delay = 0.2    # 其他連線 (src/dest)

    return prop_delay + proc_delay


def get_cost_traffic(u_type, v_type, c_base=0.02, alpha=10, beta=6):
    # Region/User 與 Cloud（地面）
    if (u_type in ("src", "dest") and v_type == "cloud") or \
       (v_type in ("src", "dest") and u_type == "cloud"):
        return c_base

    # Region/User 與 Satellite
    if (u_type in ("src", "dest") and v_type == "satellite") or \
       (v_type in ("src", "dest") and u_type == "satellite"):
        return alpha * c_base

    # Cloud 與 Cloud（地面骨幹）
    if u_type == "cloud" and v_type == "cloud":
        return c_base

    # Cloud 與 Satellite
    if (u_type == "cloud" and v_type == "satellite") or \
       (v_type == "cloud" and u_type == "satellite"):
        return alpha * c_base

    # Satellite 與 Satellite（ISL）
    if u_type == "satellite" and v_type == "satellite":
        return beta * c_base

    # 預設
    return c_base

def add_edge_with_cost(G, u, v, latency, bandwidth, c_base=0.02, alpha=10, beta=6):
    u_type = G.nodes[u]["type"]
    v_type = G.nodes[v]["type"]
    cost_traffic = get_cost_traffic(u_type, v_type, c_base, alpha, beta)
    G.add_edge(
        u, v,
        latency=latency,
        bandwidth=bandwidth,
        used_bandwidth=0,
        cost_traffic=cost_traffic
    )
    
# 小工具：取得節點在時間 t 的位置（支援「共用軌道參數」）
def get_pos(meta, t):
    if not meta["mobile"]:
        return meta["pos0"]
    if "orbit" in meta and meta["orbit"] is not None:
        inc = meta["orbit"]["inc"]
        w   = meta["orbit"]["w"]
        r   = meta["orbit"]["r"]
        phi_t = meta["phi0"] + w * t
        return spherical_to_cartesian(r, inc, phi_t)
    
    raise ValueError(f"Invalid mobile node without orbit info: {meta}")

def _sample_edge_bw(avg, rng, lower_ratio=0.4, upper_ratio=1.8):
    """
    回傳一個整數的 edge bandwidth。
    - 有 50% 機率抽在 < avg 的區間 [lower_ratio*avg, 1.0*avg)
    - 有 50% 機率抽在 > avg 的區間 (1.0*avg, upper_ratio*avg]
    """
    if not (0 < lower_ratio < 1.0 < upper_ratio):
        raise ValueError("lower_ratio 必須 < 1.0 且 upper_ratio 必須 > 1.0")
    
    if avg <= 0:
        return 1
    if rng.random() < 0.5:
        ratio = rng.uniform(lower_ratio, 1.0)   # 小於平均
    else:
        ratio = rng.uniform(1.0, upper_ratio)   # 大於平均
    return max(1, int(avg * ratio))

# === 以距離群聚 dest 成為 Regions ===
def _assign_regions_to_dests(G, dests, pos, thr):
    # 用距離門檻建一張 dest-clique 圖，連通分量就是一個 Region
    H = nx.Graph()
    H.add_nodes_from(dests)
    for i in range(len(dests)):
        for j in range(i + 1, len(dests)):
            di, dj = dests[i], dests[j]
            if euclid_latency(pos[di], pos[dj]) <= thr:
                H.add_edge(di, dj)

    regions = {}
    # 為了結果穩定，依成員名稱排序後決定 region 編號
    comps = list(nx.connected_components(H))
    comps.sort(key=lambda c: min(c))  # 依字典序最小的節點名排序
    for ridx, comp in enumerate(comps):
        rid = f"R{ridx}"
        members = sorted(list(comp))
        # 計算該 Region 的重心（平均座標）
        coords = np.array([pos[n] for n in members], dtype=float)
        centroid = tuple(coords.mean(axis=0))
        regions[rid] = {"members": members, "centroid": centroid}
        # print(f"{members}")
        # 對每個 dest 設定 region 屬性
        for n in members:
            G.nodes[n]["region"] = rid
    # 也把 region 資訊放在圖層級，方便之後查詢
    G.graph["regions"] = regions

def generate_graph_sequence_random(
    n_total=40,
    total_time=10,
    radius_sat=5.0,           # 衛星軌道半徑（視覺化尺度）
    seed=42,
    # 距離門檻（決定要不要加邊）
    thr_src_to_sat=5.5,       # src -> satellite/cloud
    thr_sat_to_dest=5.5,      # satellite/cloud -> dest
    thr_sat_to_sat=3.5,       # satellite <-> satellite（ISL）
    thr_cloud_to_cloud=4.0,   # cloud <-> cloud（terrestrial）
    thr_cloud_to_sat=5.0,
    p_extra=0.03,             # 額外隨機加邊的機率
    region_dist_thr=4.0
):
    """
    生成一組動態圖：每個 time slot 回傳一張有向圖。
    - 節點型別：src, dest, satellite, cloud
    - 至少 1 src 與 1 dest；其餘型別隨機分配
    - 衛星會依軌道移動（同軌道共用 inc/r/w；軌道內衛星相位均勻分佈）；雲端/地面點不移動（z=0）
    - 依據距離門檻自動加邊，並少量隨機補強拓撲
    - 邊會附帶 latency 屬性（以歐氏距離代替）
    """
    assert n_total >= 4, "總 vertex 數需 >= 4"
        
    rng = random.Random(seed)

    # --- 至少 1 src / 1 dest，其餘型別隨機 ---
    types_pool = ["satellite", "cloud", "dest"]
    node_types = ["satellite", "cloud", "src", "dest"]
    num_satellite = 1
    for _ in range(n_total - 4):
        t = rng.choice(types_pool)
        node_types.append(t)
        if t == "satellite":
            num_satellite += 1
    rng.shuffle(node_types)

    # --- 根據衛星總數決定軌道數，並平均分配每軌衛星數 ---
    num_orbits = int(math.floor(math.sqrt(num_satellite)))
    num_orbits = max(1, min(num_orbits, num_satellite))  # 每軌至少 1 顆
    orbit_sat_counts = [1] * num_orbits
    cnt = num_satellite - num_orbits
    for i in range(cnt):
        orbit_sat_counts[rng.randrange(0, num_orbits)] += 1

    # 建立每條軌道的共用參數與該軌道的相位表
    orbits = []
    for orbit_id, m_i in enumerate(orbit_sat_counts):
        inc = math.radians(rng.uniform(25, 155))
        w   = rng.uniform(0.02, 0.08)
        r   = radius_sat * (0.9 + 0.2 * rng.random())
        phi0_base = 2 * math.pi * rng.random()
        jitter = 0.05
        # 均勻分相 + 輕微抖動
        phases = [phi0_base + 2 * math.pi * k / m_i + rng.uniform(-jitter, jitter) for k in range(m_i)]
        orbits.append(dict(inc=inc, w=w, r=r, phases=phases, orbit_id=orbit_id))

    sat_assignments = []  # [(orbit_id, k_within_orbit), ...] 對應 sat_indices 的順序
    cnt = 0
    for o, od in enumerate(orbits):
        for k in range(len(od["phases"])):
            if cnt < num_satellite:
                sat_assignments.append((o, k))
                cnt += 1

    # --- 為每個節點指派靜態/動態參數 ---
    nodes_meta = []
    sat_assign_ptr = 0
    for idx, t in enumerate(node_types):
        name = f"v{idx}"
        if t in ("src", "dest", "cloud"):
            spread = 10.0
            x = rng.uniform(-spread, spread)
            y = rng.uniform(-spread, spread)
            z = 0.0
            meta = dict(
                name=name,
                type=t,
                mobile=False,
                orbit=None,
                pos0=(x, y, z),
                bandwidth=(rng.randint(15, 25) if t == "src" else 0)
            )
        elif t == "satellite" and num_satellite > 0:
            orbit_id, k = sat_assignments[sat_assign_ptr]
            sat_assign_ptr += 1
            od = orbits[orbit_id]
            meta = dict(
                name=name,
                type="satellite",
                mobile=True,
                orbit=dict(inc=od["inc"], w=od["w"], r=od["r"], orbit_id=orbit_id),
                phi0=od["phases"][k],
                bandwidth=0
            )
        else:
            raise ValueError(f"Unexpected node type: {t} at index {idx}")
        nodes_meta.append(meta)

    # --- 逐時刻建圖 ---
    graph_seq = []
    for t in range(total_time):
        G = nx.DiGraph(time=t)

        # 加入節點
        for meta in nodes_meta:
            p = get_pos(meta, t)
            node_type = meta["type"]
            
            # 預設值
            storage_model = None
            d = 0.0
            z_val = 0.0
            gamma = 0.0
            req_size = 0.0
            bandwidth = meta.get("bandwidth", 0)
            
            capacity = round(rng.uniform(100, 500), 2)  # 隨機容量 GB
            storage_used = 0.0
            
            if node_type == "dest":
                req_size = round(rng.uniform(5, 50), 2)

            elif node_type == "cloud":
                storage_model = "linear"
                d = round(rng.uniform(0.015, 0.03), 5)
                z_val = 1.0
                gamma = 1.0

            elif node_type == "satellite":
                storage_model = "concave"
                d = round(rng.uniform(0.02, 0.04), 5)
                z_val = 0.8
                gamma = 3.0
            
            G.add_node(
                meta["name"],
                type=node_type,
                time=t,
                pos=p,
                bandwidth=bandwidth,
                storage_model=storage_model,
                d=d,
                z=z_val,
                gamma=gamma,
                req_size=req_size,
                capacity=capacity,        
                storage_used=storage_used,
                orbit_id=(meta["orbit"]["orbit_id"] if meta.get("orbit") else None)
            )
            
            if node_type == "src":
                G.nodes[meta["name"]]["data_size"] = rng.randint(15, 25)

        # 分組
        srcs  = [n for n, d in G.nodes(data=True) if d["type"] == "src"]
        dests = [n for n, d in G.nodes(data=True) if d["type"] == "dest"]
        sats  = [n for n, d in G.nodes(data=True) if d["type"] == "satellite"]
        clouds= [n for n, d in G.nodes(data=True) if d["type"] == "cloud"]

        pos = {n: G.nodes[n]["pos"] for n in G.nodes}
        
        src_bws = [G.nodes[s]["bandwidth"] for s in srcs]
        avg_src_bw = (sum(src_bws) / len(src_bws)) if src_bws else 10  # 沒 src 時給個小的預設

        # 呼叫：把本時槽的 dest 分群
        _assign_regions_to_dests(G, dests, pos, region_dist_thr)

        # s -> satellite/cloud
        for s in srcs:
            for target in sats + clouds:
                dist_km = euclid_latency(pos[s], pos[target])  # km
                if dist_km <= thr_src_to_sat:
                    bw = _sample_edge_bw(avg_src_bw, rng)
                    latency_ms = realistic_latency(pos[s], pos[target], G.nodes[s]["type"], G.nodes[target]["type"])
                    add_edge_with_cost(G, s, target, latency_ms, bw)

        # satellite/cloud <-> d
        for dn in dests:
            for sl in sats + clouds:
                dist_km = euclid_latency(pos[sl], pos[dn])
                if dist_km <= thr_sat_to_dest:
                    bw = _sample_edge_bw(avg_src_bw, rng)
                    latency_ms = realistic_latency(pos[sl], pos[dn], G.nodes[sl]["type"], G.nodes[dn]["type"])
                    add_edge_with_cost(G, sl, dn, latency_ms, bw)
                    add_edge_with_cost(G, dn, sl, latency_ms, bw)

        # sat <-> sat (ISL)
        for i in range(len(sats)):
            for j in range(i + 1, len(sats)):
                a, b = sats[i], sats[j]
                dist_km = euclid_latency(pos[a], pos[b])
                if dist_km <= thr_sat_to_sat:
                    bw = _sample_edge_bw(avg_src_bw, rng)
                    lat_ab = realistic_latency(pos[a], pos[b], G.nodes[a]["type"], G.nodes[b]["type"])
                    lat_ba = realistic_latency(pos[b], pos[a], G.nodes[b]["type"], G.nodes[a]["type"])
                    add_edge_with_cost(G, a, b, lat_ab, bw)
                    add_edge_with_cost(G, b, a, lat_ba, bw)

        # cloud <-> cloud
        for i in range(len(clouds)):
            for j in range(i + 1, len(clouds)):
                a, b = clouds[i], clouds[j]
                dist_km = euclid_latency(pos[a], pos[b])
                if dist_km <= thr_cloud_to_cloud:
                    bw = _sample_edge_bw(avg_src_bw, rng)
                    lat_ab = realistic_latency(pos[a], pos[b], G.nodes[a]["type"], G.nodes[b]["type"])
                    lat_ba = realistic_latency(pos[b], pos[a], G.nodes[b]["type"], G.nodes[a]["type"])
                    add_edge_with_cost(G, a, b, lat_ab, bw)
                    add_edge_with_cost(G, b, a, lat_ba, bw)

        # cloud <-> satellite
        for i in range(len(clouds)):
            for j in range(len(sats)):
                a, b = clouds[i], sats[j]
                dist_km = euclid_latency(pos[a], pos[b])
                if dist_km <= thr_cloud_to_sat:
                    bw = _sample_edge_bw(avg_src_bw, rng)
                    lat_ab = realistic_latency(pos[a], pos[b], G.nodes[a]["type"], G.nodes[b]["type"])
                    lat_ba = realistic_latency(pos[b], pos[a], G.nodes[b]["type"], G.nodes[a]["type"])
                    add_edge_with_cost(G, a, b, lat_ab, bw)
                    add_edge_with_cost(G, b, a, lat_ba, bw)
        
        # 隨機補邊, 暫時用來解決DMTS出現Unreachable Dest
        nodes = list(G.nodes)
        for _ in range(int(p_extra * len(nodes) * (len(nodes) - 1))):
            u, v = rng.sample(nodes, 2)

            # 取得兩端的節點類型
            u_type = G.nodes[u]["type"]
            v_type = G.nodes[v]["type"]

            # 避開 src->dest 直連，避免自己連自己，避免已存在的邊
            if u != v and not G.has_edge(u, v) and not (u_type == "src" and v_type == "dest"):
                bw = _sample_edge_bw(avg_src_bw, rng)
                lat_ms = realistic_latency(pos[u], pos[v], G.nodes[u]["type"], G.nodes[v]["type"])
                add_edge_with_cost(G, u, v, lat_ms, bw)

        graph_seq.append(G)

    return graph_seq
        
def main():
    dir_path = "output_graphs"
    os.makedirs(dir_path, exist_ok=True)

    # 取得資料夾內的 txt 檔數量
    txt_count = len([f for f in os.listdir(dir_path) if f.endswith(".txt")])

    # seed 根據檔案數量決定
    graphs = generate_graph_sequence_random(seed=txt_count + 1, n_total=100, total_time=10)

    # print_graphs(graphs)
    
    save_graph_sequence_to_txt(graph_seq=graphs)
    
    loaded_graph_seq = load_graph_sequence_from_txt(path="output_graphs", idx=txt_count + 1)
    # print_graphs(loaded_graph_seq)
    print(are_graphs_equal(graphs, loaded_graph_seq))

if __name__ == "__main__":
    main()
