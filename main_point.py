import argparse
import os
import time
import numpy as np
import json
from datetime import date  # 用於處理日期
import yaml  # 用於讀取yml格式的配置文件 


# 確保可以從同一個 utils 目錄導入
from utils.atom_pick import determine_target_count, adaptive_threshold_analysis, parse_probabilities, nms_kdtree_adaptive,resolve_cross_class_overlaps_keep_maxprob
from utils.bond_matching import bond_match, Params
from utils.point_cal_iou import write_mrc_with_geometry, report_iou_from_files


# 取得當前腳本的目錄路徑
script_dir = os.path.dirname(os.path.abspath(__file__))

# 配置文件的路徑
config_file_path = f"{script_dir}/config/arguments_point.yml"
COMMENT_MARKER = '#'  # 註解的標誌

# 處理命令行參數
def process_arguments(args):
    if args.config is not None:
        # 讀取配置文件並過濾掉註解行
        config_dict = yaml.safe_load(args.config) # 讀取
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith(COMMENT_MARKER)} # 過濾
        args.config = args.config.name # 配置文件的路徑
    else:
        config_dict = dict() # 如果沒有提供配置文件，則創建一個空字典

    return config_dict



# 將所有流程包裝成函式，方便多次呼叫
def run_once(config_dict):
    """
    跑一次完整流程，使用指定 coverage_factor。
    tag 會用來標記檔名與子資料夾，例如 'cf0.90'
    回傳：該 coverage_factor 的所有階段 IoU rows（list[dict]）
    """
    
    rows_all = []
        
    # 1) 推估目標數
    base_target = determine_target_count(config_dict['fasta'])
    target_coverage = int(base_target * config_dict['coverage_factor'])
    print(f"\n[GRID] cf={config_dict['coverage_factor']:.2f} → target_atoms={target_coverage}")
 
    # 2) 基於自適應閾值，乘上倍率
    ca_t, n_t, c_t = adaptive_threshold_analysis(config_dict['prob_file'], target_coverage)
    def clamp01(x): return max(0.0, min(1.0, float(x)))
    
    ca_t = clamp01(ca_t * config_dict['ca_mult'])
    n_t  = clamp01(n_t  * config_dict['n_mult'])
    c_t  = clamp01(c_t  * config_dict['c_mult'])

    print(f"[GRID] thresholds  CA={ca_t:.4f} (x{config_dict['ca_mult']}), N={n_t:.4f} (x{config_dict['n_mult']}), C={c_t:.4f} (x{config_dict['c_mult']})")

    # 3) 解析概率點
    ca_pts, n_pts, c_pts = parse_probabilities(config_dict['prob_file'], ca_t, n_t, c_t)
    count_thr = {1: len(ca_pts), 2: len(n_pts), 3: len(c_pts)}
    
    # 4) NMS
    ca_nms,ca_final_r = nms_kdtree_adaptive(ca_pts,config_dict['nms_radius'], max_points=target_coverage)
    n_nms,n_final_r  = nms_kdtree_adaptive(n_pts, config_dict['nms_radius'], max_points=int(target_coverage))
    c_nms,c_final_r  = nms_kdtree_adaptive(c_pts, config_dict['nms_radius'], max_points=target_coverage)
    count_nms = {1: len(ca_nms), 2: len(n_nms), 3: len(c_nms)}

    # 5) 原子匹配
    bm_cfg = config_dict['bond_matching']
    
    if isinstance(bm_cfg, dict):
        # 從 YAML 來的是 dict，把它變成 Params dataclass
        bm_p = Params(**bm_cfg)

    ca_b, n_b, c_b = bond_match(ca_nms, n_nms, c_nms, bm_p,return_triplets=True, as_tuple=True)
    count_bonds = {1: len(ca_b), 2: len(n_b), 3: len(c_b)}
    
    # 6) 交叉類別同體素去重（保留最大 prob / CA>N>C）
    ca_f, n_f, c_f = resolve_cross_class_overlaps_keep_maxprob(ca_b, n_b, c_b, config_dict['label_mrc'])
    count_final = {1: len(ca_f), 2: len(n_f), 3: len(c_f)}

    # 7) 寫出各階段 MRC（只用於 IoU；檔名加上 tag）
    # 寫出 MRC 並計算 IoU
    tag = f"cf{config_dict['coverage_factor']:.2f}_nm{config_dict['nms_radius']:.2f}_mul{config_dict['ca_mult']:.2f}-{config_dict['n_mult']:.2f}-{config_dict['c_mult']:.2f}"
    out_dir = os.path.join(config_dict['output_dir'], tag)
    os.makedirs(out_dir, exist_ok=True)

    mrc_thr = os.path.join(out_dir, "stage_threshold.mrc")
    mrc_nms = os.path.join(out_dir, "stage_nms.mrc")
    mrc_bnd = os.path.join(out_dir, "stage_bonds.mrc")
    mrc_fin = os.path.join(out_dir, "stage_final.mrc")

    write_mrc_with_geometry(ca_pts, n_pts, c_pts, config_dict['label_mrc'], mrc_thr)
    write_mrc_with_geometry(ca_nms, n_nms, c_nms, config_dict['label_mrc'], mrc_nms)
    write_mrc_with_geometry(ca_b, n_b, c_b, config_dict['label_mrc'], mrc_bnd)
    write_mrc_with_geometry(ca_f, n_f, c_f, config_dict['label_mrc'], mrc_fin)

    # 8) 計算 IoU（每階段）
    at_cm, at_per_class, at_macro_f1, at_macro_iou, at_ap_per_class, at_map = report_iou_from_files("After Thresholding", config_dict['label_mrc'], mrc_thr, ca_prob=ca_pts, n_prob=n_pts, c_prob=c_pts, out_path=out_dir, R_sphere=config_dict['r_sphere'])
    an_cm, an_per_class, an_macro_f1, an_macro_iou, an_ap_per_class, an_map = report_iou_from_files("After NMS", config_dict['label_mrc'], mrc_nms, ca_prob=ca_nms, n_prob=n_nms, c_prob=c_nms, out_path=out_dir, R_sphere=config_dict['r_sphere'])
    au_cm, au_per_class, au_macro_f1, au_macro_iou, au_ap_per_class, au_map = report_iou_from_files("After Bonds Match", config_dict['label_mrc'], mrc_bnd, ca_prob=ca_b, n_prob=n_b, c_prob=c_b, out_path=out_dir, R_sphere=config_dict['r_sphere'])
    af_cm, af_per_class, af_macro_f1, af_macro_iou, af_ap_per_class, af_map = report_iou_from_files("Final", config_dict['label_mrc'], mrc_fin, ca_prob=ca_f, n_prob=n_f, c_prob=c_f, out_path=out_dir, R_sphere=config_dict['r_sphere'])

    final_avg_iou =  np.round(af_macro_iou, 4)
    trial_record = {
        "Date" : str(date.today()),
        "Version": config_dict['version'],
        "EMD_ID": config_dict['emd_id'],
        "R_distance":  config_dict['r_sphere'],
        "coverage_factor": config_dict['coverage_factor'],
        "ca_multiplier": config_dict['ca_mult'],
        "n_multiplier": config_dict['n_mult'],
        "c_multiplier": config_dict['c_mult'],
        "nms":{
            "org_nms_radius": config_dict['nms_radius'],
            "ca_final_nms_r": ca_final_r,
            "n_final_nms_r": n_final_r,
            "c_final_nms_r": c_final_r,
        },
        "bond_params": {  
            "r_ca_c_max": bm_p.r_ca_c_max,
            "r_ca_n_max": bm_p.r_ca_n_max,
            "r_ca_c_low": bm_p.r_ca_c_low ,
            "r_ca_n_low": bm_p.r_ca_n_low ,
            "use_angle": bm_p.use_angle,
            "angle_target": bm_p.angle_target,
            "angle_tol": bm_p.angle_tol,
            "w_angle": bm_p.w_angle,
            "bondlen_window": bm_p.bondlen_window,
            "ca_c_len_lo": bm_p.ca_c_len_lo, "ca_c_len_hi": bm_p.ca_c_len_hi,
            "ca_n_len_lo": bm_p.ca_n_len_lo, "ca_n_len_hi": bm_p.ca_n_len_hi,
            "distance_power": bm_p.distance_power,
            "w_dist_c": bm_p.w_dist_c, "w_dist_n": bm_p.w_dist_n,
            "exclusive_match": bm_p.exclusive_match,
        },
        "counts": {
            "After Thresholding": count_thr,
            "After NMS": count_nms,
            "After Bonds": count_bonds,
            "Final": count_final,
        },
        "CA_atom_iou_metrics": {
            "After Thresholding": at_per_class["CA"],
            "After NMS": an_per_class["CA"],
            "After Bonds": au_per_class["CA"],
            "Final": af_per_class["CA"],
        },
        "N_atom_iou_metrics": {
            "After Thresholding": at_per_class["N"],
            "After NMS": an_per_class["N"],
            "After Bonds": au_per_class["N"],
            "Final": af_per_class["N"],
        },
        "C_atom_iou_metrics": {
            "After Thresholding": at_per_class["C"],
            "After NMS": an_per_class["C"],
            "After Bonds": au_per_class["C"],
            "Final": af_per_class["C"],
        },
        "confusion_matrix": {
            "After Thresholding": at_cm,
            "After NMS": an_cm,
            "After Bonds": au_cm,
            "Final": af_cm,
        },
        "AP_per_class": {
            "After Thresholding": at_ap_per_class,
            "After NMS": an_ap_per_class,
            "After Bonds": au_ap_per_class,
            "Final": af_ap_per_class,
        },
        "avg_iou_metrics": {
            "After Thresholding": at_macro_iou,
            "After NMS": an_macro_iou,
            "After Bonds": au_macro_iou,
            "Final": af_macro_iou,
        },
        "avg_f1_metrics": {
            "After Thresholding": at_macro_f1,
            "After NMS": an_macro_f1,
            "After Bonds": au_macro_f1,
            "Final": af_macro_f1,
        },
        "avg_AP_metrics": {
            "After Thresholding": at_map,
            "After NMS": an_map,
            "After Bonds": au_map,
            "Final": af_map,
        },
        "final_avg_iou": np.round(af_macro_iou, 4),
        "final_f1": np.round(af_macro_f1, 4),
        "final_map": np.round(af_map, 4),
        "out_dir": out_dir,
        "final_mrc": mrc_fin
    }
    best_json = os.path.join(config_dict['output_dir'], config_dict['json_name'])
    with open(best_json, "w") as f:
        json.dump(trial_record, f, indent=2)
    
    return final_avg_iou, trial_record, rows_all

#############################################################################################


def build_parser():
    p = argparse.ArgumentParser(description="原子聚類優化")

    # YAML 設定檔（可選）
    p.add_argument('--config', type=argparse.FileType(mode='r'),
                        default=config_file_path)

    p.add_argument("--stage_prefix", default=None, help="階段檔名前綴（預設沿用 --output 前綴）")
      
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    config_dict = process_arguments(args)
    
    if args.stage_prefix is None:
        args.stage_prefix = f"{config_dict['emd_id']}"

    # 建立總 stage_dir
    os.makedirs(config_dict['output_dir'], exist_ok=True)
        
    # 列印當前參數
    print("#"*20, "Running with below configuration","#"*20)
    for k, v in config_dict.items():
        print(k, "=", v)
        
    # 單一因子
    print(f"\n========== Running cf{config_dict['coverage_factor']:.2f} ==========")
    final_avg_iou, trial_record, rows_all = run_once(config_dict)
    print(f"\n[RESULT] final_avg_iou = {final_avg_iou}")
           

if __name__ == "__main__":
    print("Date:",date.today())
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"總執行時間: {elapsed_time:.2f} 秒")
    
# python3 -u main_point.py > 14147.log 2>&1