import numpy as np
from scipy.spatial import cKDTree


def ultra_permissive_chemical_bonds(ca_centroids, n_centroids, c_centroids):
    """
    超寬鬆化學鍵結驗證 - 優先保留更多原子
    """
    print(f"\n=== 超寬鬆化學鍵結驗證 ===")
    if not ca_centroids or not c_centroids:
        print("警告: CA或C質心列表為空")
        return ca_centroids, n_centroids, c_centroids
    
    if not n_centroids:
        print("警告: N質心列表為空，返回所有CA和C")
        return ca_centroids, [], c_centroids

    ca_coords = np.array([c[:3] for c in ca_centroids])
    n_coords = np.array([n[:3] for n in n_centroids])
    c_coords = np.array([c[:3] for c in c_centroids])

    ca_tree = cKDTree(ca_coords)
    c_tree = cKDTree(c_coords)
    n_tree = cKDTree(n_coords)
    
    # 步驟1: 寬鬆的CA-C配對 (距離上限5.0Å)
    ca_c_pairs = {}
    ca_used = set()
    c_used = set()
    
    for i, ca_coord in enumerate(ca_coords):
        if i in ca_used:
            continue
            
        nearby_c_indices = c_tree.query_ball_point(ca_coord, r=5.0)
        best_c_idx = None
        min_dist = float('inf')
        
        for c_idx in nearby_c_indices:
            if c_idx not in c_used:
                dist = np.linalg.norm(ca_coord - c_coords[c_idx])
                if dist < min_dist:
                    min_dist = dist
                    best_c_idx = c_idx
        
        if best_c_idx is not None:
            ca_c_pairs[i] = best_c_idx
            ca_used.add(i)
            c_used.add(best_c_idx)
    
    # 步驟2: 超寬鬆的N原子配對 (距離上限6.0Å，無角度限制)
    ca_n_pairs = {}
    n_used = set()
    
    for ca_idx in ca_c_pairs.keys():
        ca_coord = ca_coords[ca_idx]
        nearby_n_indices = n_tree.query_ball_point(ca_coord, r=6.0)
        
        best_n_idx = None
        min_dist = float('inf')
        
        for n_idx in nearby_n_indices:
            if n_idx not in n_used:
                dist = np.linalg.norm(ca_coord - n_coords[n_idx])
                if dist < min_dist:
                    min_dist = dist
                    best_n_idx = n_idx
        
        if best_n_idx is not None:
            ca_n_pairs[ca_idx] = best_n_idx
            n_used.add(best_n_idx)
    
    # 組裝最終結果
    final_ca = [ca_centroids[i] for i in ca_c_pairs.keys()]
    final_c = [c_centroids[ca_c_pairs[i]] for i in ca_c_pairs.keys()]
    final_n = [n_centroids[ca_n_pairs[i]] for i in ca_c_pairs.keys() if i in ca_n_pairs]
    
    print(f"配對結果: CA-C配對 {len(final_ca)} 個, N-CA配對 {len(final_n)} 個")
    print(f"N/CA比率: {len(final_n)/len(final_ca) if final_ca else 0:.2f}")
    
    return final_ca, final_n, final_c