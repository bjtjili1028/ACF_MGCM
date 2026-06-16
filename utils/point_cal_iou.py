import numpy as np
import math
from typing import List, Dict
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree  # <--- 引入加速神器

####################################################################################################
# === 只給 IoU 用：用 label_mrc 幾何寫出 0/1/2/3 的 MRC ===
# 輸出各階段的 label_mrc
def write_mrc_with_geometry(ca_items, n_items, c_items, geom_mrc_path, out_path):
    """
    ca_items / n_items / c_items 可是 (x,y,z,p) tuple 或具有 .x,.y,.z,.prob 的物件
    以 geom_mrc 的 shape/origin/voxel_size 映射到體素，1=CA, 2=N, 3=C
    """
    def as_xyzp(it):
        if hasattr(it, "x") and hasattr(it, "y") and hasattr(it, "z"):
            return float(it.x), float(it.y), float(it.z), float(getattr(it, "prob", 1.0))
        x, y, z = float(it[0]), float(it[1]), float(it[2])
        p = float(it[3]) if len(it) >= 4 else 1.0
        return x, y, z, p

    with mrcfile.open(geom_mrc_path, mode='r') as m:
        shape = m.data.shape
        # origin
        try:
            ox, oy, oz = float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)
        except Exception:
            o = m.header.origin; ox, oy, oz = float(o[0]), float(o[1]), float(o[2])
        # voxel
        try:
            vx, vy, vz = float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)
        except Exception:
            vs = m.voxel_size; vx, vy, vz = float(vs[0]), float(vs[1]), float(vs[2])

    vol = np.zeros(shape, dtype=np.int32)

    def place(items, cid):
        if not items: return 0
        Z, Y, X = vol.shape
        cnt = 0
        for it in items:
            x, y, z, _ = as_xyzp(it)
            ix = int(round((x - ox) / vx))
            iy = int(round((y - oy) / vy))
            iz = int(round((z - oz) / vz))
            if 0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X:
                vol[iz, iy, ix] = cid
                cnt += 1
        return cnt

    c1 = place(ca_items, 1)
    c2 = place(n_items, 2)
    c3 = place(c_items, 3)

    with mrcfile.open(geom_mrc_path, mode='r') as mref, mrcfile.new(out_path, overwrite=True) as mout:
        mout.set_data(vol.astype(np.float32))
        # 繼承幾何
        try:
            mout.voxel_size = (mref.voxel_size.x, mref.voxel_size.y, mref.voxel_size.z)
        except Exception:
            mout.voxel_size = mref.voxel_size
        try:
            mout.header.origin = (mref.header.origin.x, mref.header.origin.y, mref.header.origin.z)
        except Exception:
            mout.header.origin = mref.header.origin

    print(f"[stage out] {out_path} | placed CA={c1}, N={c2}, C={c3}")


def write_mrc_with_geometry_CA(ca_items, geom_mrc_path, out_path):
    """
    ca_items / n_items / c_items 可是 (x,y,z,p) tuple 或具有 .x,.y,.z,.prob 的物件
    以 geom_mrc 的 shape/origin/voxel_size 映射到體素，1=CA, 2=N, 3=C
    """
    def as_xyzp(it):
        if hasattr(it, "x") and hasattr(it, "y") and hasattr(it, "z"):
            return float(it.x), float(it.y), float(it.z), float(getattr(it, "prob", 1.0))
        x, y, z = float(it[0]), float(it[1]), float(it[2])
        p = float(it[3]) if len(it) >= 4 else 1.0
        return x, y, z, p

    with mrcfile.open(geom_mrc_path, mode='r') as m:
        shape = m.data.shape
        # origin
        try:
            ox, oy, oz = float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)
        except Exception:
            o = m.header.origin; ox, oy, oz = float(o[0]), float(o[1]), float(o[2])
        # voxel
        try:
            vx, vy, vz = float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)
        except Exception:
            vs = m.voxel_size; vx, vy, vz = float(vs[0]), float(vs[1]), float(vs[2])

    vol = np.zeros(shape, dtype=np.int32)

    def place(items, cid):
        if not items: return 0
        Z, Y, X = vol.shape
        cnt = 0
        for it in items:
            x, y, z, _ = as_xyzp(it)
            ix = int(round((x - ox) / vx))
            iy = int(round((y - oy) / vy))
            iz = int(round((z - oz) / vz))
            if 0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X:
                vol[iz, iy, ix] = cid
                cnt += 1
        return cnt

    c1 = place(ca_items, 1)
    # c2 = place(n_items, 2)
    # c3 = place(c_items, 3)

    with mrcfile.open(geom_mrc_path, mode='r') as mref, mrcfile.new(out_path, overwrite=True) as mout:
        mout.set_data(vol.astype(np.float32))
        # 繼承幾何
        try:
            mout.voxel_size = (mref.voxel_size.x, mref.voxel_size.y, mref.voxel_size.z)
        except Exception:
            mout.voxel_size = mref.voxel_size
        try:
            mout.header.origin = (mref.header.origin.x, mref.header.origin.y, mref.header.origin.z)
        except Exception:
            mout.header.origin = mref.header.origin

    print(f"[stage out] {out_path} | placed CA={c1}")
####################################################################################################
# === 以檔案為單位計算 IoU/Precision/Recall，並印出結果 ===
# 讀取 MRC 及獲取其幾何資訊
def read_atom_label_mrc(path: str):
    """
    讀取「label 版本」的 MRC 檔：
    - data: 3D int32 array, 0=背景, 1/2/3=不同原子類型
    - header_info: 包含 origin、voxel_size、shape
    """
    with mrcfile.open(path, mode="r") as m:
        data = m.data.copy().astype(np.int32)  # (Z, Y, X)

        # 讀取 origin（有些 MRC 用 header.origin.x/y/z，有些是 array）
        try:
            ox, oy, oz = float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)
        except Exception:
            ox, oy, oz = [float(v) for v in m.header.origin]

        # 讀取 voxel_size（同樣可能是屬性或 array）
        try:
            vx, vy, vz = float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)
        except Exception:
            vx, vy, vz = [float(v) for v in m.voxel_size]

    header_info = {
        "origin": {"x": ox, "y": oy, "z": oz},
        "voxel_size": {"x": vx, "y": vy, "z": vz},
        "shape": data.shape,
    }
    return data, header_info

# 將座標轉換回資料格式
def extract_atoms_from_label(
    label_vol: np.ndarray,
    header_info: dict,
    id_to_name: Dict[int, str] = None,
    default_score: float = 1.0,
):
    """
    從 label volume 中抽出「原子清單」。

    輸入：
    - label_vol[z, y, x] ∈ {0,1,2,3,...}
      0  = 背景
      1  = Cα
      2  = N
      3  = C
      ... 其他 id 如有需要可在 id_to_name 裡加上對應
    - header_info: 由 read_atom_label_mrc 回傳，包含 origin & voxel_size
    - id_to_name: 類別 id → 類別名稱（字串）
    - default_score: 無機率時，給每顆 GT 原子一個固定 score（AP 不會用到 GT 的 score）

    回傳：
    - atoms: List[dict]，每個 dict 為：
        {
          "xyz": np.array([x, y, z], float),   # 世界座標（Å）
          "cls": "Cα" / "N" / "C",
          "score": float                       # 這裡 GT 一律是 default_score
        }
    """
    if id_to_name is None:
        id_to_name = {1: "CA", 2: "N", 3: "C"}

    # 找出所有非 0 voxel
    mask = label_vol > 0
    zs, ys, xs = np.where(mask)
    cids = label_vol[zs, ys, xs].astype(int)

    # 取出 voxel_size 和 origin，用來從 index → 物理座標（Å）
    vx = header_info["voxel_size"]["x"]
    vy = header_info["voxel_size"]["y"]
    vz = header_info["voxel_size"]["z"]
    ox = header_info["origin"]["x"]
    oy = header_info["origin"]["y"]
    oz = header_info["origin"]["z"]

    # index 轉世界座標：
    # x_world = origin_x + ix * voxel_size_x
    xs_w = ox + xs * vx
    ys_w = oy + ys * vy
    zs_w = oz + zs * vz

    xyzs = np.stack([xs_w, ys_w, zs_w], axis=1).astype(np.float32)

    atoms = []
    for xyz, cid in zip(xyzs, cids):
        if cid in id_to_name:
            atoms.append({
                "xyz": xyz,
                "cls": id_to_name[cid],
                "score": float(default_score),
            })

    return atoms


# 讀取各階段的原子座標及機率輸出成 atoms 格式 for AP
def prob_points_to_atoms(points, cls_name: str, default_score: float = 1.0):
    """
    將「機率點」列表轉成評估程式共用的 atoms 格式：
      [{"xyz": np.array([x,y,z], float),
        "cls": cls_name,
        "score": prob}, ...]

    兼容以下格式：
      1. 物件有 .x, .y, .z, .prob   (e.g. WeightedPoint, WPoint)
      2. tuple/list: (x, y, z, prob)
      3. tuple/list: ((x, y, z), prob)

    說明：
    - 這樣你不論是：
        ca_points (WeightedPoint)
        ca_final_points (WPoint)
      或是自訂 tuple，通通可以轉成同一種 atoms 格式，後續
      point-based / sphere-based / AP 的程式都可以重用。
    """
    atoms = []
    for p in points:
        # Case 1: 物件有屬性 x, y, z, prob
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            x = float(p.x)
            y = float(p.y)
            z = float(p.z)
            score = float(getattr(p, "prob", default_score))

        # Case 2 & 3: tuple / list
        else:
            # 例如 (x, y, z, prob)
            if len(p) == 4 and not isinstance(p[0], (list, tuple, np.ndarray)):
                x, y, z, score = map(float, p)

            # 例如 ((x, y, z), prob)
            elif len(p) == 2:
                (x, y, z) = p[0]
                x = float(x); y = float(y); z = float(z)
                score = float(p[1])

            else:
                raise ValueError(f"不支援的點格式: {p}")

        atoms.append({
            "xyz": np.array([x, y, z], dtype=np.float32),
            "cls": cls_name,
            "score": score,
        })

    return atoms

# 讀取各階段的原子座標及機率輸出成 atoms 格式 for ALL
def quantize_prob_atoms(points, cls_name, geom_mrc_path, prob_num):
    """
    對 WeightedPoint / (x,y,z,p) 的連續座標套用：
        index = round((x - origin) / voxel_size)
        x_q   = origin + index * voxel_size
    讓幾何與 MRC label system 完全一致。
    score(機率) 保留不變，用於 AP。
    """
    import mrcfile
    import numpy as np

    # --- 1. 從 geom_mrc_path 讀出 origin + voxel_size ---
    with mrcfile.open(geom_mrc_path, mode='r') as m:
        # origin
        try:
            ox, oy, oz = float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)
        except:
            ox, oy, oz = [float(v) for v in m.header.origin]

        # voxel_size
        try:
            vx, vy, vz = float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)
        except:
            vx, vy, vz = [float(v) for v in m.voxel_size]

    atoms = []

    # --- 2. 對每一個「機率點」做量化 ---
    for p in points:
        # 2-1. 讀出 (x,y,z,prob)
        if hasattr(p, "x"):
            # WeightedPoint / WPoint 類型
            x, y, z = float(p.x), float(p.y), float(p.z)
            score = float(getattr(p, "prob", prob_num))
        else:
            # tuple / list
            if len(p) == 4:
                # (x, y, z, prob)
                x, y, z, score = map(float, p)
            else:
                # ((x,y,z), prob)
                (x, y, z), score = p
                x, y, z, score = float(x), float(y), float(z), float(score)

        # 2-2. 套用「與 write_mrc_with_geometry 一樣的 index 計算」
        ix = int(round((x - ox) / vx))
        iy = int(round((y - oy) / vy))
        iz = int(round((z - oz) / vz))

        # 2-3. 再用 extract_atoms_from_label 的反向公式轉回來
        x_q = ox + ix * vx
        y_q = oy + iy * vy
        z_q = oz + iz * vz

        # 2-4. 存成統一 atoms 格式（保留 score）
        atoms.append({
            "xyz": np.array([x_q, y_q, z_q], dtype=np.float32),  # ← 已量化座標
            "cls": cls_name,
            "score": score,                                      # ← 原機率
        })

    return atoms


# 混淆矩陣 → per-class 指標（IoU / Precision / Recall / Specificity / F1）
def metrics_from_confusion(
    cm: Dict[str, Dict[str, int]],
    classes: List[str],
    total_gt: int,
):
    """
    根據 confusion matrix 計算：
    - 每類的 TP, FP, FN, TN
    - Precision, Recall, Specificity, F1, IoU
    - Macro F1、Macro IoU
    """
    per_class = {}

    for cls in classes:
        # 真正例：真實 = cls，預測 = cls
        tp = cm[cls][cls]

        # 假正例：預測為 cls，但真實不是 cls
        fp = 0
        for t in classes:
            if t != cls:
                fp += cm[t][cls]   # 真實 t，被錯預測成 cls
        fp += cm["FP"][cls]         # 來自「憑空出現」的預測（沒有對應任何 GT）

        # 假負例：真實是 cls，但預測成其他類 / 沒預測到
        fn = 0
        for p in classes:
            if p != cls:
                fn += cm[cls][p]   # 真實 cls，被錯預測成其他類
        fn += cm[cls]["FN"]        # 真實 cls，完全沒被預測到

        # 真負例：不是 cls 的那些 GT
        neg = total_gt - (tp + fn)
        tn = neg - fp if neg - fp >= 0 else 0

        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou         = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[cls] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1=f1,
            iou=iou,
        )

    macro_f1  = sum(per_class[c]["f1"]  for c in classes) / len(classes)
    macro_iou = sum(per_class[c]["iou"] for c in classes) / len(classes)

    return per_class, macro_f1, macro_iou

def print_confusion_matrix(cm: Dict[str, Dict[str, int]], classes: List[str]):
    """
    以較好閱讀的格式印出混淆矩陣。

    行（True）: classes + ['FP']   # 'FP' 代表「沒有 GT，純預測出來的點」
    列（Pred）: classes + ['FN']   # 'FN' 代表「有 GT，但預測不到」

    - cm[真實類][預測類]
    - cm['FP'][cls_pred] = 假正例，純預測出來沒有配到任何 GT 的數量
    - cm[cls_true]['FN'] = 假負例，真實是該類但完全沒有被預測到
    """
    pred_headers = classes + ["FN"]
    true_headers = classes + ["FP"]

    print("=== Confusion Matrix ===")
    print("True \\ Pred".ljust(12) + " | " + " | ".join([f"{h:^8}" for h in pred_headers]))
    print("-" * (14 + 11 * len(pred_headers)))

    for t in true_headers:
        row = f"{t:<12} | "
        vals = []
        for p in pred_headers:
            if t == "FP" and p == "FN":
                vals.append(f"{'N/A':^8}")
            else:
                vals.append(f"{cm[t][p]:^8}")
        print(row + " | ".join(vals))
    print()

# 物件偵測 - 點對點版本 Point-based 物件偵測評估（距離門檻 R_match）
def eval_point_based(
    gt_atoms: List[dict],
    pred_atoms: List[dict],
    classes: List[str],
    R_match: float = 2.0,
):
    """
    以「點對點距離」來決定 TP/FP/FN：

    - 對每一個 GT atom，去找「距離最近」的 pred atom
    - 若距離 <= R_match，視為匹配（TP），否則視為 FN
    - 每個 pred atom 最多只能被配對一次
    - 沒有被配對到的 pred atom 視為 FP

    回傳：
    - cm: confusion matrix
    - per_class: 每一類的指標（precision / recall / specificity / F1 / IoU）
    - macro_f1: 所有類別 F1 的平均
    - macro_iou: 所有類別 IoU 的平均
    """
    # 初始化混淆矩陣
    cm = {t: {p: 0 for p in classes + ["FN"]} for t in classes + ["FP"]}

    used_pred = set()

    # --- 先處理所有 GT（決定 TP / FN + 類別匹配狀況）---
    for g in gt_atoms:
        g_cls = g["cls"]
        g_xyz = g["xyz"]

        best_d = float("inf")
        best_j = -1

        # 找最近的 pred，且尚未被使用
        for j, p in enumerate(pred_atoms):
            if j in used_pred:
                continue
            d = float(np.linalg.norm(p["xyz"] - g_xyz))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j != -1 and best_d <= R_match:
            # 有找到且距離在門檻內 → 視為成功匹配
            p_cls = pred_atoms[best_j]["cls"]
            cm[g_cls][p_cls] += 1
            used_pred.add(best_j)
        else:
            # 找不到 or 距離太遠 → FN
            cm[g_cls]["FN"] += 1

    # --- 再處理所有 pred 中「未匹配」的 → FP ---
    for j, p in enumerate(pred_atoms):
        if j not in used_pred:
            p_cls = p["cls"]
            cm["FP"][p_cls] += 1

    total_gt = len(gt_atoms)
    per_class, macro_f1, macro_iou = metrics_from_confusion(cm, classes, total_gt)

    return cm, per_class, macro_f1, macro_iou



#### new_ap 
def compute_strict_ap_point_based(
    gt_atoms: List[dict],
    pred_atoms: List[dict],
    classes: List[str],
    R_match: float,
    out_path,  # 為了保持介面一致保留參數，即使這裡沒用到
):
    """
    計算【嚴格版】AP：
    在判定 TP 時，必須確保該預測點「最近的 GT」確實是「同類別」。
    如果最近的 GT 是異類別，則視為誤判 (FP)，即使 R_match 範圍內有同類別 GT 也不行。
    """
    ap_per_class = {}

    # 1. 預先處理：建立所有 GT 的空間索引 (或是簡單列表)，方便查找「最近的原子」
    # 為了簡單與通用，我們這裡用暴力法找最近 (因為有點對點距離計算)
    # 如果效能有問題，這裡可以用 cKDTree 加速，但邏輯是一樣的
    
    for cls in classes:
        # 取出該類別的所有預測 (我們要算這個類別的 PR 曲線)
        pred_cls_subset = [p for p in pred_atoms if p["cls"] == cls]
        
        # 如果該類別沒有預測，AP = 0
        if len(pred_cls_subset) == 0:
            ap_per_class[cls] = 0.0
            continue
            
        # 依信心分數排序
        pred_cls_subset.sort(key=lambda x: x["score"], reverse=True)
        
        # 取出該類別的 GT 數量 (用於計算 Recall 分母)
        n_gt_target_class = sum(1 for g in gt_atoms if g["cls"] == cls)
        
        if n_gt_target_class == 0:
            ap_per_class[cls] = 0.0
            continue

        tp_list = []
        fp_list = []
        
        # 記錄哪些 GT 已經被配對掉了 (避免重複配對)
        # key: GT在原始列表中的 index
        gt_matched_indices = set()

        # 對每一個預測點 (依高分到低分)
        for p in pred_cls_subset:
            p_xyz = p["xyz"]
            
            # --- 嚴格邏輯核心 ---
            # 尋找「所有 GT (不分由類別)」中最近的那一個
            best_dist = float("inf")
            best_gt_idx = -1
            
            for i, g in enumerate(gt_atoms):
                # 如果這個 GT 已經被更高分的預測拿走了，就略過
                if i in gt_matched_indices:
                    continue
                
                dist = float(np.linalg.norm(g["xyz"] - p_xyz))
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = i
            
            # 判定結果
            if best_gt_idx != -1 and best_dist <= R_match:
                # 找到了最近的 GT，檢查類別是否吻合
                matched_gt = gt_atoms[best_gt_idx]
                
                if matched_gt["cls"] == cls:
                    # 距離夠近，且類別正確 -> TP
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_matched_indices.add(best_gt_idx)
                else:
                    # 距離夠近，但類別錯誤 (撞到 N 或 C 了) -> FP (嚴格扣分!)
                    # 原本的 lenient AP 會忽略這個 N，繼續往遠處找 CA，但在這裡不行
                    tp_list.append(0)
                    fp_list.append(1)
                    # 注意：這裡通常策略是「誤判也算佔用了這個 GT」，或者「不佔用」
                    # 在 Object Detection 中通常誤判就是純 FP。
                    # 為了最嚴格邏輯，既然判錯了，我們就標記為 FP。
            else:
                # 附近完全沒有任何 GT -> FP (背景雜訊)
                tp_list.append(0)
                fp_list.append(1)

        # --- 計算 AP 積分 ---
        tp_arr = np.array(tp_list, dtype=np.float32)
        fp_arr = np.array(fp_list, dtype=np.float32)
        
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        recalls = tp_cum / n_gt_target_class
        
        # PASCAL VOC 插值法
        recalls = np.concatenate([[0.0], recalls])
        precisions = np.concatenate([[precisions.max() if len(precisions)>0 else 0.0], precisions])
        
        # 確保單調遞減
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        ap = 0.0
        for i in range(1, len(recalls)):
            dr = recalls[i] - recalls[i - 1]
            if dr > 0:
                ap += precisions[i] * dr
                
        ap_per_class[cls] = ap

    mAP = sum(ap_per_class.values()) / len(classes)
    return ap_per_class, mAP

def compute_ap_point_based(
    gt_atoms: List[dict],
    pred_atoms: List[dict],
    classes: List[str],
    R_match: float,
    out_path,
):
    """
    在所有類別上計算 AP（Point-based）：
    現在改為直接呼叫嚴格版 (Strict) 計算邏輯。
    """
    ap_per_class, mAP = compute_strict_ap_point_based(
        gt_atoms,
        pred_atoms,
        classes,
        R_match,
        out_path
    )

    return ap_per_class, mAP

def plot_strict_f1_and_ap_curves(gt_atoms, pred_atoms, classes, R_match, out_dir):
    """
    極速版繪圖函式 (使用 cKDTree 加速)：
    1. [Strict F1]: 預先計算鄰居，秒殺 100 次門檻掃描。
    2. [Standard PR]: 維持標準 mAP 邏輯。
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"=== 開始繪製圖表 (R_match={R_match}) ===")
    plt.style.use('seaborn-whitegrid')

    # ---------------------------------------------------------
    # 準備數據 (轉成 numpy array 以利加速)
    # ---------------------------------------------------------
    print(">>> [加速預處理] 正在建立 KD-Tree 與預計算鄰居...")
    
    if len(pred_atoms) == 0:
        print("警告：沒有預測點，跳過繪圖。")
        return

    # 提取 Pred 資訊
    pred_xyz = np.array([p['xyz'] for p in pred_atoms])
    pred_scores = np.array([p['score'] for p in pred_atoms])
    pred_cls_names = np.array([p['cls'] for p in pred_atoms])
    
    # 提取 GT 資訊
    gt_xyz = np.array([g['xyz'] for g in gt_atoms])
    gt_cls_names = np.array([g['cls'] for g in gt_atoms])

    # ---------------------------------------------------------
    # 步驟 1: 預先計算所有 GT 的潛在匹配對象 (只做一次！)
    # ---------------------------------------------------------
    # 建立 Pred 的空間索引
    tree = cKDTree(pred_xyz)
    
    # 一次性找出所有 GT 在 R_match 範圍內的鄰居
    # query_results 是一個列表，每個元素是該 GT 附近的 pred indices
    neighbor_indices_list = tree.query_ball_point(gt_xyz, R_match)

    # 整理每個 GT 的候選人列表 (按距離排序)
    # gt_candidates[i] = [(dist, pred_idx), (dist, pred_idx)...]
    gt_candidates = []
    for i, indices in enumerate(neighbor_indices_list):
        if len(indices) == 0:
            gt_candidates.append([])
            continue
        
        # 算出該 GT 與這些鄰居的距離
        neighbors_xyz = pred_xyz[indices]
        dists = np.linalg.norm(neighbors_xyz - gt_xyz[i], axis=1)
        
        # 打包並依照距離排序 (最近的優先)
        # 格式: (pred_global_index, distance)
        candidates = sorted(zip(indices, dists), key=lambda x: x[1])
        gt_candidates.append(candidates)

    print(f">>> 預處理完成，開始執行 101 次門檻掃描...")

    # ---------------------------------------------------------
    # Part 1: 極速計算 "嚴格 F1 曲線"
    # ---------------------------------------------------------
    thresholds = np.linspace(0.0, 1.0, 101)
    results = {c: {'th': [], 'f1': []} for c in classes}

    for th in thresholds:
        # 1. 快速標記哪些 Pred 通過門檻 (Boolean Mask)
        passed_mask = (pred_scores >= th)
        
        # 統計通過門檻的各類別總數 (用於快速計算 FP)
        # passed_counts = {'CA': 100, 'N': 80...}
        passed_counts = {c: 0 for c in classes}
        unique, counts = np.unique(pred_cls_names[passed_mask], return_counts=True)
        for u, c in zip(unique, counts):
            passed_counts[u] = c
            
        # 2. 模擬混淆矩陣
        cm = {t: {p: 0 for p in classes + ["FN"]} for t in classes}
        # 用來記錄哪些 pred 已經被 GT 認領了
        used_pred_indices = set()
        
        # 3. 遍歷每個 GT，直接查表 (不用再算距離了！)
        for i, candidates in enumerate(gt_candidates):
            g_cls = gt_cls_names[i]
            
            match_found = False
            # 在預先算好的鄰居裡，找第一個「分數夠高」且「沒人用過」的
            for pred_idx, dist in candidates:
                if passed_mask[pred_idx] and (pred_idx not in used_pred_indices):
                    # 找到了！(Greedy Match)
                    p_cls = pred_cls_names[pred_idx]
                    cm[g_cls][p_cls] += 1
                    used_pred_indices.add(pred_idx)
                    match_found = True
                    break
            
            if not match_found:
                cm[g_cls]["FN"] += 1

        # 4. 快速計算 FP (背景雜訊)
        # 背景 FP = (該類別所有通過門檻的點) - (該類別被 GT 拿走的點)
        # 注意：被 GT 拿走的點，可能是 TP (同類)，也可能是誤判 (不同類)，反正都不是背景 FP
        cm["FP"] = {}
        for cls in classes:
            # 算出有多少 cls 類別的預測點被使用了
            used_count_of_cls = sum(1 for idx in used_pred_indices if pred_cls_names[idx] == cls)
            cm["FP"][cls] = passed_counts.get(cls, 0) - used_count_of_cls

        # 5. 計算 F1
        for cls in classes:
            tp = cm[cls][cls]
            # FP (嚴格版) = 背景雜訊 + 別人誤判成我
            fp = cm["FP"][cls] + sum(cm[t][cls] for t in classes if t != cls)
            # FN (嚴格版) = 漏抓 + 我誤判成別人
            fn = cm[cls]["FN"] + sum(cm[cls][p] for p in classes if p != cls)
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            results[cls]['th'].append(th)
            results[cls]['f1'].append(f1)

    # 繪圖 (Strict F1)
    for cls in classes:
        ths = np.array(results[cls]['th'])
        f1s = np.array(results[cls]['f1'])
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_th = ths[best_idx]
        curr_f1 = f1s[0]

        plt.figure(figsize=(10, 6))
        plt.plot(ths, f1s, label=f'{cls} Strict F1', color='tab:blue', linewidth=2)
        plt.plot(best_th, best_f1, 'ro', label=f'Best: {best_f1:.2f} @ {best_th:.2f}')
        plt.plot(0.0, curr_f1, 'go', label=f'Start (Th=0): {curr_f1:.2f}')
        
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Strict F1 Score (Report Logic)')
        plt.title(f'Strict F1 vs Threshold - Class: {cls}\n(Accounts for Misclassification)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 1.0); plt.ylim(0, 1.0)
        
        save_path = os.path.join(out_dir, f'Strict_F1_{cls}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[{cls}] 嚴格 F1 圖: {save_path} (Best: {best_th:.2f})")

    # ---------------------------------------------------------
    # Part 2: 繪製標準 PR 曲線 (mAP) - 這裡維持原樣，不需要特別優化
    # ---------------------------------------------------------
    print(">>> 正在計算標準 PR 曲線 (mAP Logic)...")
    # 為了不破壞原本的邏輯，這裡還是用列表生成式重跑一次簡單版
    # (因為 One-vs-Rest 計算比較單純，資料量切分後跑起來很快)
    for cls in classes:
        gt_cls_indices = [i for i, c in enumerate(gt_cls_names) if c == cls]
        pred_cls_indices = [i for i, c in enumerate(pred_cls_names) if c == cls]
        n_gt = len(gt_cls_indices)
        
        if n_gt == 0: continue
        
        # 取出該類別的資料
        curr_pred_scores = pred_scores[pred_cls_indices]
        curr_pred_xyz = pred_xyz[pred_cls_indices]
        curr_gt_xyz = gt_xyz[gt_cls_indices]
        
        # 排序
        sort_idx = np.argsort(curr_pred_scores)[::-1]
        curr_pred_xyz = curr_pred_xyz[sort_idx]
        
        tp_list = []; fp_list = []; gt_matched = np.zeros(n_gt, dtype=bool)
        
        for p_pos in curr_pred_xyz:
            # 這裡資料量變少(單一類別)，可以直接算
            dists = np.linalg.norm(curr_gt_xyz - p_pos, axis=1)
            best_idx = np.argmin(dists) if len(dists) > 0 else -1
            min_dist = dists[best_idx] if best_idx != -1 else float('inf')
            
            if best_idx != -1 and min_dist <= R_match and not gt_matched[best_idx]:
                tp_list.append(1); fp_list.append(0)
                gt_matched[best_idx] = True
            else:
                tp_list.append(0); fp_list.append(1)
        
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / n_gt
        
        plt.figure(figsize=(8, 8))
        plt.plot(recalls, precisions, label=f'{cls} PR Curve', color='tab:orange', linewidth=2)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Standard PR Curve - Class: {cls}')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 1.0); plt.ylim(0, 1.0)
        
        save_path = os.path.join(out_dir, f'PR_Curve_{cls}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[{cls}] PR 曲線圖: {save_path}")

    print("=== 繪圖完成 ===\n")

####################################################################################################
# === 以檔案為單位計算 IoU/Precision/Recall，並印出結果 ===
def report_iou_from_files(stage_name, label_file, pred_file, ca_prob, n_prob, c_prob, out_path, R_sphere = 4.2):
    """
    以檔案為單位計算 IoU/Precision/Recall（只算 1/2/3 類；同時印出 CA/N/C 名稱）
    回傳：
      class_metrics: {1|2|3: {'iou','precision','recall'}, ...}
      avg_metrics:   {'iou','precision','recall'}  # 僅 1~3 的平均
    """
    
    id_to_name = {1: "CA", 2: "N", 3: "C"}
    classes = ["CA", "N", "C"]
    
    gt_vol, gt_info = read_atom_label_mrc(label_file)
    gt_atoms = extract_atoms_from_label(gt_vol, gt_info, id_to_name=id_to_name, default_score=1.0)
    
    # pred_vol,  pred_info = read_atom_label_mrc(pred_file)
    # pred_atoms = extract_atoms_from_label(pred_vol, pred_info, id_to_name=id_to_name, default_score=0.0)
    
    pred_atoms = []
    pred_atoms += quantize_prob_atoms(ca_prob, "CA", pred_file, prob_num=0)
    pred_atoms += quantize_prob_atoms(n_prob,  "N",  pred_file, prob_num=0)
    pred_atoms += quantize_prob_atoms(c_prob,  "C",  pred_file, prob_num=0)

    # 加入機率點
    # pred_atoms = prob_points_to_atoms(ca_prob, "CA", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(n_prob, "N", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(c_prob, "C", default_score=0.0)
    
    # --- Sphere-based IoU / F1 等指標 ---
    cm, per_class, macro_f1, macro_iou = eval_point_based(
        gt_atoms,
        pred_atoms,
        classes,
        R_match=R_sphere,
    )

    print(f"\n=== Point-Match-based 評估 @ {stage_name} ===")
    print(f"  R_match = {R_sphere}")
    print_confusion_matrix(cm, classes)

    print("[Per-class 指標]")
    for cls in classes:
        m = per_class[cls]
        print(
            f"  {cls}: "
            f"IoU={m['iou']:.4f}, "
            f"PRECISION={m['precision']:.4f}, "
            f"RECALL={m['recall']:.4f}, "
            f"Spec={m['specificity']:.4f}, "
            f"F1={m['f1']:.4f}"
        )
    print(f"[Macro] F1={macro_f1:.4f}, IoU={macro_iou:.4f}")

    # # 加入機率點
    # pred_atoms = prob_points_to_atoms(ca_prob, "CA", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(n_prob, "N", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(c_prob, "C", default_score=0.0)
    
    # --- Sphere-based AP / mAP ---
    ap_per_cls, mAP = compute_ap_point_based(
        gt_atoms,
        pred_atoms,
        classes,
        R_match=R_sphere,
        out_path=f"{out_path}/{stage_name}_"
    )
    
    print("\n[AP (Point-Match-based)]")
    for cls in classes:
        print(f"  {cls}: AP={ap_per_cls[cls]:.4f}")
    print(f"mAP (Point-Match-based): {mAP:.4f}\n")

    # plot_strict_f1_and_ap_curves(
    #     gt_atoms, 
    #     pred_atoms, 
    #     classes, 
    #     R_match=R_sphere, 
    #     out_dir=out_path  # 圖片會存到這個資料夾
    # )

    return cm, per_class, macro_f1, macro_iou, ap_per_cls, mAP


####################################################################################################
# for CA
def report_iou_from_files_CA(stage_name, label_file, pred_file, ca_prob, out_path, R_sphere = 4.2):
    """
    以檔案為單位計算 IoU/Precision/Recall（只算 1 類；印出 CA 名稱）
    回傳：
      class_metrics: {1|2|3: {'iou','precision','recall'}, ...}
    """
    
    id_to_name = {1: "CA"}
    classes = ["CA"]
    
    gt_vol, gt_info = read_atom_label_mrc(label_file)
    gt_atoms = extract_atoms_from_label(gt_vol, gt_info, id_to_name=id_to_name, default_score=1.0)
    
    # pred_vol,  pred_info = read_atom_label_mrc(pred_file)
    # pred_atoms = extract_atoms_from_label(pred_vol, pred_info, id_to_name=id_to_name, default_score=0.0)
    
    pred_atoms = []
    pred_atoms += quantize_prob_atoms(ca_prob, "CA", pred_file, prob_num=0)
    # pred_atoms += quantize_prob_atoms(n_prob,  "N",  pred_file, prob_num=0)
    # pred_atoms += quantize_prob_atoms(c_prob,  "C",  pred_file, prob_num=0)

    # 加入機率點
    # pred_atoms = prob_points_to_atoms(ca_prob, "CA", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(n_prob, "N", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(c_prob, "C", default_score=0.0)
    
    # --- Sphere-based IoU / F1 等指標 ---
    cm, per_class, macro_f1, macro_iou = eval_point_based(
        gt_atoms,
        pred_atoms,
        classes,
        R_match=R_sphere,
    )

    print(f"\n=== Point-Match-based 評估 @ {stage_name} ===")
    print(f"  R_match = {R_sphere}")
    print_confusion_matrix(cm, classes)

    print("[Per-class 指標]")
    for cls in classes:
        m = per_class[cls]
        print(
            f"  {cls}: "
            f"IoU={m['iou']:.4f}, "
            f"PRECISION={m['precision']:.4f}, "
            f"RECALL={m['recall']:.4f}, "
            f"Spec={m['specificity']:.4f}, "
            f"F1={m['f1']:.4f}"
        )
    print(f"[Macro] F1={macro_f1:.4f}, IoU={macro_iou:.4f}")

    # # 加入機率點
    # pred_atoms = prob_points_to_atoms(ca_prob, "CA", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(n_prob, "N", default_score=0.0)
    # pred_atoms += prob_points_to_atoms(c_prob, "C", default_score=0.0)
    
    # --- Sphere-based AP / mAP ---
    ap_per_cls, mAP = compute_ap_point_based(
        gt_atoms,
        pred_atoms,
        classes,
        R_match=R_sphere,
        out_path=f"{out_path}/{stage_name}_"
    )
    
    print("\n[AP (Point-Match-based)]")
    for cls in classes:
        print(f"  {cls}: AP={ap_per_cls[cls]:.4f}")
    print(f"mAP (Point-Match-based): {mAP:.4f}\n")

    # plot_strict_f1_and_ap_curves(
    #     gt_atoms, 
    #     pred_atoms, 
    #     classes, 
    #     R_match=R_sphere, 
    #     out_dir=out_path  # 圖片會存到這個資料夾
    # )

    return cm, per_class, macro_f1, macro_iou, ap_per_cls, mAP