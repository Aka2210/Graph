import argparse
import math
import os
import random
import numpy as np
import networkx as nx
from Save_And_Read_Graphs import save_graph_sequence_to_txt
from Save_And_Read_Graphs import load_graph_sequence_from_txt
from Debug import are_graphs_equal
# from Save_And_Read_Graphs import save_graph_sequence_to_txt
# from Save_And_Read_Graphs import load_graph_sequence_from_txt
# from Debug import print_graphs
# from Debug import are_graphs_equal

# --- 真實模型所需的新增常數 ---

# 地球半徑 (km)
EARTH_RADIUS_KM = 6371.0
# 地對空連線的最小仰角 (degrees)
MIN_ELEVATION_ANGLE = 15.0 

# --- 真實模型所需的新增輔助函數 ---

def get_elevation_angle(p_ground, p_sat):
    """
    計算衛星相對於地面站的仰角 (degrees)
    p_ground 和 p_sat 都是 (x,y,z) 地心座標
    """
    # 確保 p_ground 在地球表面上 (或至少有一個範數)
    norm_p_ground = np.linalg.norm(p_ground)
    if norm_p_ground == 0:
        return -90.0 # 無法計算

    # 向量：從地面站指向上方 (本地 Z 軸 / 天頂)
    vec_zenith = np.array(p_ground) / norm_p_ground
    
    # 向量：從地面站指向衛星
    vec_gnd_to_sat = np.array(p_sat) - np.array(p_ground)
    norm_vec_gnd_to_sat = np.linalg.norm(vec_gnd_to_sat)
    if norm_vec_gnd_to_sat == 0:
        return 90.0 # 衛星就在地面站上

    # 計算天頂角 (Zenith Angle)
    dot_product = np.dot(vec_zenith, vec_gnd_to_sat)
    
    # 避免 acos 範圍錯誤
    cos_zenith_angle = max(-1.0, min(1.0, dot_product / norm_vec_gnd_to_sat))
    
    zenith_angle_rad = np.arccos(cos_zenith_angle)
    
    # 仰角 = 90 度 - 天頂角
    elevation_rad = (np.pi / 2.0) - zenith_angle_rad
    return np.degrees(elevation_rad)

def is_earth_blocking(p1, p2, earth_radius=EARTH_RADIUS_KM):
    """
    檢查 p1 和 p2 之間的直線路徑是否被地球遮擋
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # 向量 v = p2 - p1
    v = p2 - p1
    norm_v_sq = np.dot(v, v)
    if norm_v_sq == 0:
        return False # 兩點重合

    # 向量 u = -p1 (從 p1 指向原點)
    u = -p1
    
    # 計算 p1 到 (p1p2連線的最近原點) 的投影長度比例
    t = np.dot(u, v) / norm_v_sq
    
    # 如果最近點在 p1 和 p2 之間 (0 <= t <= 1)
    if 0 <= t <= 1:
        # 計算最近點
        closest_point = p1 + t * v
        # 如果最近點在地球半徑內，則被遮擋
        if np.linalg.norm(closest_point) < earth_radius:
            return True
            
    # 如果最近點不在線段上，我們只需要檢查兩個端點
    # (但此處假設 p1, p2 都在地球外，所以此檢查可省略)
    return False

def generate_walker_meta(
    num_sats_total=60, 
    num_planes=6, 
    altitude_km=550.0, 
    inclination_deg=53.0,
    f_phasing_param=0,
    base_angular_velocity=0.05
):
    """
    生成 Walker 星座的元數據 (meta)
    """
    
    if num_sats_total % num_planes != 0:
        raise ValueError("衛星總數必須能被軌道平面數整除")
        
    sats_per_plane = num_sats_total // num_planes
    radius_km = EARTH_RADIUS_KM + altitude_km
    inclination_rad = math.radians(inclination_deg)
    
    satellites_meta = []
    
    for p_idx in range(num_planes):
        # 1. 計算軌道平面的經度 (RAAN)
        raan_rad = math.radians(p_idx * (360.0 / num_planes))
        
        for k_idx in range(sats_per_plane):
            # 2. 計算衛星在軌道內的初始相位 (Phase)
            phase_in_plane = math.radians(k_idx * (360.0 / sats_per_plane))
            
            # 3. 應用 F 參數進行平面間相位偏移
            phase_offset = math.radians(f_phasing_param * (p_idx * 360.0 / num_sats_total))
            
            phi0 = phase_in_plane + phase_offset
            
            meta = dict(
                # 'name' 將在主函數中分配
                type="satellite",
                mobile=True,
                orbit=dict(
                    r=radius_km, 
                    inc=inclination_rad, 
                    raan=raan_rad,          # 軌道平面的經度
                    w=base_angular_velocity, # 軌道角速度
                    orbit_id=p_idx,       # 軌道平面 ID
                    id_in_plane=k_idx     # 在軌道內的 ID
                ),
                phi0=phi0,
            )
            satellites_meta.append(meta)
            
    return satellites_meta

# --- 保留的輔助函數 (稍作修改或原封不動) ---

def euclid_latency(p1, p2):
    # 此函數現在計算的是 3D 地心座標系中的直線距離 (km)
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
    # (此函數原封不動)
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
    # (此函數原封不動)
    u_type = G.nodes[u]["type"]
    v_type = G.nodes[v]["type"]
    cost_traffic = get_cost_traffic(u_type, v_type, c_base, alpha, beta)
    G.add_edge(
        u, v,
        latency=latency,
        bandwidth=bandwidth,
        used_bandwidth=0,
        cost_traffic=cost_traffic,
        virtual=False
    )
    
# --- 座標計算函數 (已替換為真實軌道模型) ---

def get_pos(meta, t):
    """
    小工具：取得節點在時間 t 的位置（支援「共用軌道參數」）
    使用 ECEF (地心) 座標系
    """
    if not meta["mobile"]:
        return meta["pos0"]
    
    if "orbit" in meta and meta["orbit"] is not None:
        orb = meta["orbit"]
        r   = orb["r"]
        inc = orb["inc"]
        raan= orb["raan"] # 軌道平面經度
        w   = orb["w"]
        
        # 1. 計算衛星在軌道平面上的當前相位
        phi_t = meta["phi0"] + w * t
        
        # 2. 計算在 2D 軌道平面上的座標 (z=0)
        x_orb = r * np.cos(phi_t)
        y_orb = r * np.sin(phi_t)
        
        # 3. 依據軌道傾角 (inc) 進行旋轉 (繞 X 軸)
        x_rot_i = x_orb
        y_rot_i = y_orb * np.cos(inc)
        z_rot_i = y_orb * np.sin(inc)
        
        # 4. 依據 RAAN (raan) 進行旋轉 (繞 Z 軸)
        x = x_rot_i * np.cos(raan) - y_rot_i * np.sin(raan)
        y = x_rot_i * np.sin(raan) + y_rot_i * np.cos(raan)
        z = z_rot_i
        
        return x, y, z
    
    raise ValueError(f"Invalid mobile node without orbit info: {meta}")

# --- 保留的輔助函數 ---

def _sample_edge_bw(avg, rng, lower_ratio=0.4, upper_ratio=1.8, min_bw=10):
    """
    回傳一個整數的 edge bandwidth。
    在 [lower_ratio*avg, upper_ratio*avg] 區間內隨機抽樣，並保證至少 min_bw。
    """
    if lower_ratio > upper_ratio:
        raise ValueError(f"lower_ratio ({lower_ratio}) 必須 <= upper_ratio ({upper_ratio})")
    
    if avg <= 0:
        return min_bw
    
    ratio = rng.uniform(lower_ratio, upper_ratio)
    return max(min_bw, int(avg * ratio))

def _assign_regions_to_dests(G, dests, pos, thr):
    # (此函數原封不動，但傳入的 thr 意義變為 km)
    H = nx.Graph()
    H.add_nodes_from(dests)
    for i in range(len(dests)):
        for j in range(i + 1, len(dests)):
            di, dj = dests[i], dests[j]
            if euclid_latency(pos[di], pos[dj]) <= thr:
                H.add_edge(di, dj)
    regions = {}
    comps = list(nx.connected_components(H))
    comps.sort(key=lambda c: min(c))
    for ridx, comp in enumerate(comps):
        rid = f"R{ridx}"
        members = sorted(list(comp))
        coords = np.array([pos[n] for n in members], dtype=float)
        centroid = tuple(coords.mean(axis=0))
        regions[rid] = {"members": members, "centroid": centroid}
        for n in members:
            G.nodes[n]["region"] = rid
    G.graph["regions"] = regions
    
# --- 核心函數：生成圖形序列 (已大幅修改) ---

def generate_graph_sequence_realistic(
    # 節點數量
    n_sats=60,
    n_clouds=10,
    n_srcs=5,
    n_dests=25,
    # 時間
    total_time=10,
    seed=42,
    # Walker 星座參數
    num_planes=6,           
    altitude_km=550.0,
    inclination_deg=53.0,
    f_phasing_param=0,
    base_angular_velocity=0.05,
    # 地面連線參數
    thr_cloud_to_cloud=2000.0,
    region_dist_thr=1000.0
):
    """
    生成一組基於真實 Walker 星座模型的動態圖。
    - 節點型別：src, dest, satellite, cloud
    - 衛星依 Walker 星座佈局並移動
    - 地面站 (src, dest, cloud) 隨機分佈於地球表面
    - 地對空連線：基於最小仰角
    - 空對空連線 (ISL)：基於結構化鄰居與地球遮擋
    """
    
    if n_sats % num_planes != 0:
        print(f"警告: 衛星數 {n_sats} 無法被平面數 {num_planes} 整除。")
        sats_per_plane = math.ceil(n_sats / num_planes)
        n_sats = sats_per_plane * num_planes
        print(f"修正後衛星總數為: {n_sats}")

    rng = random.Random(seed)
    np.random.seed(seed)
        
    # === 建立節點元資料 ===
    nodes_meta = []
    
    if n_sats > 0:
        sat_metas_list = generate_walker_meta(
            num_sats_total=n_sats,
            num_planes=num_planes,
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            f_phasing_param=f_phasing_param,
            base_angular_velocity=base_angular_velocity
        )
    else:
        sat_metas_list = []
        
    sat_meta_iter = iter(sat_metas_list)

    node_types = (["satellite"] * n_sats + 
                  ["cloud"] * n_clouds + 
                  ["src"] * n_srcs + 
                  ["dest"] * n_dests)
    rng.shuffle(node_types)

    for idx, n_type in enumerate(node_types):
        name = f"v{idx}"
        if n_type == "satellite":
            meta = next(sat_meta_iter)
            meta["name"] = name
            meta["bandwidth"] = 0
        elif n_type in ("src", "dest", "cloud"):
            lat_rad = math.radians(rng.uniform(-70, 70))
            lon_rad = math.radians(rng.uniform(-180, 180))
            x = EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
            y = EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
            z = EARTH_RADIUS_KM * np.sin(lat_rad)
            meta = dict(
                name=name,
                type=n_type,
                mobile=False,
                orbit=None,
                pos0=(x, y, z),
                bandwidth=(rng.randint(15, 25) if n_type == "src" else 0)
            )
        nodes_meta.append(meta)

    # === 生成每個時間步 ===
    graph_seq = []
    for t in range(total_time):
        G = nx.DiGraph(time=t)
        for meta in nodes_meta:
            p = get_pos(meta, t)
            node_type = meta["type"]
            storage_model = None
            d = 0.0
            z_val = 0.0
            gamma = 0.0
            req_size = 0.0
            bandwidth = meta.get("bandwidth", 0)
            cache = False
            capacity = round(rng.uniform(100, 500), 2)
            storage_used = 0.0
            if node_type == "dest":
                req_size = round(rng.uniform(5, 50), 2)
            elif node_type == "cloud":
                storage_model = "linear"
                d = round(rng.uniform(0.015, 0.03), 5)
                z_val = 1.0
                gamma = 1.0
                cache = rng.choices([True, False], weights=[0.3, 0.7])[0]
            elif node_type == "satellite":
                storage_model = "concave"
                d = round(rng.uniform(0.02, 0.04), 5)
                z_val = 0.8
                gamma = 3.0
                cache = rng.choices([True, False], weights=[0.3, 0.7])[0]
            
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
                orbit_id=(meta["orbit"]["orbit_id"] if meta.get("orbit") else None),
                orbit_id_in_plane=(meta["orbit"]["id_in_plane"] if meta.get("orbit") else None),
                cache=cache
            )
            if node_type == "src":
                G.nodes[meta["name"]]["data_size"] = 1
        
        srcs  = [n for n, d in G.nodes(data=True) if d["type"] == "src"]
        dests = [n for n, d in G.nodes(data=True) if d["type"] == "dest"]
        sats  = [n for n, d in G.nodes(data=True) if d["type"] == "satellite"]
        clouds= [n for n, d in G.nodes(data=True) if d["type"] == "cloud"]
        pos = {n: G.nodes[n]["pos"] for n in G.nodes}
        
        src_bws = [G.nodes[s]["bandwidth"] for s in srcs]
        avg_src_bw = (sum(src_bws) / len(src_bws)) if src_bws else 10
        _assign_regions_to_dests(G, dests, pos, region_dist_thr)

        # === 地對空連線 ===
        ground_stations = srcs + dests + clouds
        for gnd_node in ground_stations:
            for sat_node in sats:
                if get_elevation_angle(pos[gnd_node], pos[sat_node]) > MIN_ELEVATION_ANGLE:
                    bw = _sample_edge_bw(avg_src_bw * 20, rng, 0.8, 1.2)
                    lat_gs = realistic_latency(pos[gnd_node], pos[sat_node], G.nodes[gnd_node]["type"], G.nodes[sat_node]["type"])
                    add_edge_with_cost(G, gnd_node, sat_node, lat_gs, bw)
                    lat_sg = realistic_latency(pos[sat_node], pos[gnd_node], G.nodes[sat_node]["type"], G.nodes[gnd_node]["type"])
                    add_edge_with_cost(G, sat_node, gnd_node, lat_sg, bw)

        # === 衛星 ↔ 衛星 (ISL) ===
        sats_by_plane = {}
        for s in sats:
            orb_id = G.nodes[s]["orbit_id"]
            sats_by_plane.setdefault(orb_id, []).append(s)
        sorted_plane_ids = sorted(sats_by_plane.keys())
        num_planes_actual = len(sorted_plane_ids)

        for p_idx_key, p_id in enumerate(sorted_plane_ids):
            plane_sats = sorted(sats_by_plane[p_id], key=lambda n: G.nodes[n]["orbit_id_in_plane"])
            num_sats_in_plane = len(plane_sats)
            for k_idx, sat_a in enumerate(plane_sats):
                sat_b_intra = plane_sats[(k_idx + 1) % num_sats_in_plane]
                if not G.has_edge(sat_a, sat_b_intra) and not is_earth_blocking(pos[sat_a], pos[sat_b_intra]):
                    bw = _sample_edge_bw(avg_src_bw * 50, rng, 0.8, 1.2)
                    lat_ab = realistic_latency(pos[sat_a], pos[sat_b_intra], "satellite", "satellite")
                    lat_ba = realistic_latency(pos[sat_b_intra], pos[sat_a], "satellite", "satellite")
                    add_edge_with_cost(G, sat_a, sat_b_intra, lat_ab, bw)
                    add_edge_with_cost(G, sat_b_intra, sat_a, lat_ba, bw)
                
                if num_planes_actual > 1:
                    adj_plane_id = sorted_plane_ids[(p_idx_key + 1) % num_planes_actual]
                    adj_plane_sats = sorted(sats_by_plane[adj_plane_id], key=lambda n: G.nodes[n]["orbit_id_in_plane"])
                    if k_idx < len(adj_plane_sats):
                        sat_b_inter = adj_plane_sats[k_idx]
                        if not G.has_edge(sat_a, sat_b_inter) and not is_earth_blocking(pos[sat_a], pos[sat_b_inter]):
                            bw = _sample_edge_bw(avg_src_bw * 50, rng, 0.8, 1.2)
                            lat_ab = realistic_latency(pos[sat_a], pos[sat_b_inter], "satellite", "satellite")
                            lat_ba = realistic_latency(pos[sat_b_inter], pos[sat_a], "satellite", "satellite")
                            add_edge_with_cost(G, sat_a, sat_b_inter, lat_ab, bw)
                            add_edge_with_cost(G, sat_b_inter, sat_a, lat_ba, bw)

        # === Cloud ↔ Cloud ===
        for i in range(len(clouds)):
            for j in range(i + 1, len(clouds)):
                a, b = clouds[i], clouds[j]
                dist_km = euclid_latency(pos[a], pos[b])
                if dist_km <= thr_cloud_to_cloud:
                    bw = _sample_edge_bw(avg_src_bw * 10, rng, 0.8, 1.2)
                    lat_ab = realistic_latency(pos[a], pos[b], "cloud", "cloud")
                    lat_ba = realistic_latency(pos[b], pos[a], "cloud", "cloud")
                    add_edge_with_cost(G, a, b, lat_ab, bw)
                    add_edge_with_cost(G, b, a, lat_ba, bw)

        # === Src ↔ Cloud ===
        for s in srcs:
            for c in clouds:
                dist_km = euclid_latency(pos[s], pos[c])
                if dist_km <= thr_cloud_to_cloud:
                    bw = _sample_edge_bw(avg_src_bw * 5, rng, 0.8, 1.2)
                    lat_sc = realistic_latency(pos[s], pos[c], "src", "cloud")
                    lat_cs = realistic_latency(pos[c], pos[s], "cloud", "src")
                    add_edge_with_cost(G, s, c, lat_sc, bw)
                    add_edge_with_cost(G, c, s, lat_cs, bw)
                    
        backbone_nodes = sats + clouds
        
        if not backbone_nodes and (len(srcs) > 0 or len(dests) > 0):
            # 這種情況下，沒有骨幹，src/dest 永遠無法連線
            print(f"[t={t}] 警告: 圖中沒有任何 backbone 節點 (sats/clouds)。")
        
        elif backbone_nodes: # 只有在有骨幹節點時才執行
            
            # --- 1. 強制 Src 連線 ---
            for s in srcs:
                # 檢查 Src 是否有任何對外的邊
                if G.out_degree(s) == 0:
                    # 找出最近的骨幹節點 (無論是 sat 還是 cloud)
                    closest_bb_node = min(backbone_nodes, key=lambda n: euclid_latency(pos[s], pos[n]))
                    bb_type = G.nodes[closest_bb_node]['type']
                    
                    # print(f"[t={t}] 警告: Src {s} 是孤島。強制連線到最近的 {bb_type} 節點 {closest_bb_node}。")
                    
                    # (使用一個合理的頻寬)
                    bw = _sample_edge_bw(avg_src_bw * 5, rng, 0.8, 1.2)
                    
                    lat_s_bb = realistic_latency(pos[s], pos[closest_bb_node], "src", bb_type)
                    lat_bb_s = realistic_latency(pos[closest_bb_node], pos[s], bb_type, "src")
                    
                    # 建立雙向生命線
                    add_edge_with_cost(G, s, closest_bb_node, lat_s_bb, bw)
                    add_edge_with_cost(G, closest_bb_node, s, lat_bb_s, bw)

            # --- 2. 強制 Dest 連線 ---
            for d in dests:
                # 檢查 Dest 是否已連到任何骨幹節點
                is_connected = False
                # 必須檢查雙向
                for neighbor in G.successors(d):
                    if neighbor in backbone_nodes:
                        is_connected = True
                        break
                if is_connected: continue
                
                for neighbor in G.predecessors(d):
                    if neighbor in backbone_nodes:
                        is_connected = True
                        break
                if is_connected: continue

                # 如果 Dest 還是孤島
                # 找出最近的骨幹節點
                closest_bb_node = min(backbone_nodes, key=lambda n: euclid_latency(pos[d], pos[n]))
                bb_type = G.nodes[closest_bb_node]['type']
                
                # print(f"[t={t}] 警告: Dest {d} 是孤島。強制連線到最近的 {bb_type} 節點 {closest_bb_node}。")

                bw = _sample_edge_bw(avg_src_bw * 5, rng, 0.8, 1.2)
                
                lat_d_bb = realistic_latency(pos[d], pos[closest_bb_node], "dest", bb_type)
                lat_bb_d = realistic_latency(pos[closest_bb_node], pos[d], bb_type, "dest")
                
                # 建立雙向生命線
                add_edge_with_cost(G, d, closest_bb_node, lat_d_bb, bw)
                add_edge_with_cost(G, closest_bb_node, d, lat_bb_d, bw)

        graph_seq.append(G)
        
    return graph_seq
        
def main():
    dir_path = "output_graphs"
    os.makedirs(dir_path, exist_ok=True)
    # 取得資料夾內的 txt 檔數量
    txt_count = len([f for f in os.listdir(dir_path) if f.endswith(".txt")])
    
    # 使用新的函數和參數
    graphs = generate_graph_sequence_realistic(
        seed=txt_count + 1,
        # 節點數量
        n_sats=100,         # Starlink 早期階段
        n_clouds=30,
        n_srcs=1,
        n_dests=100,
        # 時間
        total_time=10,
        # Walker 星座參數
        num_planes=6,      # 6 個軌道平面
        altitude_km=550.0, # 550 km 高度
        inclination_deg=53.0, # 53 度傾角
        f_phasing_param=1, # 簡易相位參數
        base_angular_velocity=0.06, # 角速度
        # 地面連線參數
        thr_cloud_to_cloud=3000.0, # 地面連線門檻 (km)
        region_dist_thr=1000.0    # 聚類門檻 (km)
    )
    
    # --- 以下替換為簡單的日誌輸出 ---
    print(f"成功生成 {len(graphs)} 個時間步的圖形。")
    if graphs:
        G0 = graphs[0]
        print(f"\n--- 圖形 t=0 資訊 ---")
        print(f"節點總數: {G0.number_of_nodes()}")
        print(f"邊總數: {G0.number_of_edges()}")
        
        n_sats = len([n for n, d in G0.nodes(data=True) if d['type'] == 'satellite'])
        n_clouds = len([n for n, d in G0.nodes(data=True) if d['type'] == 'cloud'])
        n_srcs = len([n for n, d in G0.nodes(data=True) if d['type'] == 'src'])
        n_dests = len([n for n, d in G0.nodes(data=True) if d['type'] == 'dest'])
        
        print(f"  - 衛星 (Satellites): {n_sats}")
        print(f"  - 地面雲 (Clouds): {n_clouds}")
        print(f"  - 源站 (Sources): {n_srcs}")
        print(f"  - 目的地 (Destinations): {n_dests}")
        
        # 檢查 ISL 連線
        isl_edges = len([
            (u, v) for u, v, d in G0.edges(data=True) 
            if G0.nodes[u]['type'] == 'satellite' and G0.nodes[v]['type'] == 'satellite'
        ])
        print(f"衛星間連線 (ISL) 總數: {isl_edges}")

        # 檢查地對空連線
        gts_edges = len([
            (u, v) for u, v, d in G0.edges(data=True) 
            if G0.nodes[u]['type'] != 'satellite' and G0.nodes[v]['type'] == 'satellite'
        ])
        print(f"地對空連線 (Gnd-to-Sat) 總數: {gts_edges}")

    # (註解掉原有的保存和讀取)
    save_graph_sequence_to_txt(graph_seq=graphs)
    loaded_graph_seq = load_graph_sequence_from_txt(path="output_graphs", idx=txt_count + 1)
    print(are_graphs_equal(graphs, loaded_graph_seq))

if __name__ == "__main__":
    main()