import argparse
import os
import time
import numpy as np
import sys
import glob
import shutil
import itertools
import json
import optuna
from typing import Optional
from datetime import date  # 用於處理日期
from functools import reduce
from operator import mul
from dataclasses import replace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 確保可以從同一個 utils 目錄導入
from utils.atom_pick import determine_target_count, adaptive_threshold_analysis, parse_probabilities, nms_kdtree_adaptive,resolve_cross_class_overlaps_keep_maxprob
from utils.bond_matching import bond_match, Params
from utils.point_cal_iou import write_mrc_with_geometry, report_iou_from_files

# 超參數主要搜索流程
def run_pipeline_with_params(args,
                             coverage_factor: float,
                             ca_mult: float,
                             n_mult: float,
                             c_mult: float,
                             nms_radius: float,
                             bm_params: Optional[Params] = None,
                             tag_prefix: str = "optuna"):
    """
    在給定參數下跑一次完整流程；回傳：
      - final_avg_iou: float（Final 階段 avg(1–3) IoU）
      - trial_record: dict（本次所有設定與數值）
      - rows_all: list[dict]（各階段 IoU rows，給後續合併輸出）
    """
    rows_all = []

    # 1) 推估目標數
    cf_start_time = time.time()
    base_target = determine_target_count(args.fasta)
    target_coverage = int(base_target * coverage_factor)
    print(f"\n[GRID] cf={coverage_factor:.2f} → target_atoms={target_coverage}")
    print(f"推估目標數執行時間: {(time.time() - cf_start_time) :.2f} 秒")


    # 2) 基於自適應閾值，乘上倍率
    adp_start_time = time.time()
    ca_t, n_t, c_t = adaptive_threshold_analysis(args.prob_file, target_coverage)
    def clamp01(x): return max(0.0, min(1.0, float(x)))
    
    ca_t = clamp01(ca_t * ca_mult)
    n_t  = clamp01(n_t  * n_mult)
    c_t  = clamp01(c_t  * c_mult)

    print(f"[GRID] thresholds  CA={ca_t:.4f} (x{ca_mult}), N={n_t:.4f} (x{n_mult}), C={c_t:.4f} (x{c_mult})")
    print(f"計算自適應閾值，乘上倍率: {(time.time() - adp_start_time) :.2f} 秒")

    # 3) 解析概率點
    use_adp_start_time = time.time()
    ca_pts, n_pts, c_pts = parse_probabilities(args.prob_file, ca_t, n_t, c_t)
    count_thr = {1: len(ca_pts), 2: len(n_pts), 3: len(c_pts)}
    n_ca, n_n, n_c = len(ca_pts), len(n_pts), len(c_pts)
    if n_ca == 0 or n_c == 0 or n_n == 0:     # 可以設更嚴格的下界
        raise optuna.exceptions.TrialPruned("Empty candidates after thresholding.")
    print(f"使用自適應閾值: {(time.time() - use_adp_start_time) :.2f} 秒")

    # 4) NMS
    nms_start_time = time.time()
    ca_nms,ca_final_r = nms_kdtree_adaptive(ca_pts,nms_radius, max_points=target_coverage)
    n_nms,n_final_r  = nms_kdtree_adaptive(n_pts, nms_radius, max_points=int(target_coverage))
    c_nms,c_final_r  = nms_kdtree_adaptive(c_pts, nms_radius, max_points=target_coverage)
    count_nms = {1: len(ca_nms), 2: len(n_nms), 3: len(c_nms)}
    print(f"nms: {(time.time() - nms_start_time) :.2f} 秒")

    # 5) 原子匹配
    bond_start_time = time.time()
    if bm_params is None:
        bm_params = Params()
    
    bm_p = replace(
        bm_params,
        r_ca_c_max=max(1e-6, bm_params.r_ca_c_max),
        r_ca_n_max=max(1e-6, bm_params.r_ca_n_max),
        angle_tol=max(1e-6, bm_params.angle_tol),
    )
    
    ca_b, n_b, c_b = bond_match(ca_nms, n_nms, c_nms, bm_p,return_triplets=True, as_tuple=True)
    count_bonds = {1: len(ca_b), 2: len(n_b), 3: len(c_b)}
    print(f"bond: {(time.time() - bond_start_time) :.2f} 秒")

    # 6) 交叉類別同體素去重（保留最大 prob / CA>N>C)
    ca_f, n_f, c_f = resolve_cross_class_overlaps_keep_maxprob(ca_b, n_b, c_b, args.label_mrc)
    count_final = {1: len(ca_f), 2: len(n_f), 3: len(c_f)}

    # 7) 寫出各階段 MRC（只用於 IoU；檔名加上 tag)
    tag = f"{tag_prefix}_cf{coverage_factor:.2f}_nm{nms_radius:.2f}_mul{ca_mult:.2f}-{n_mult:.2f}-{c_mult:.2f}"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    mrc_thr = os.path.join(out_dir, "stage_threshold.mrc")
    mrc_nms = os.path.join(out_dir, "stage_nms.mrc")
    mrc_bnd = os.path.join(out_dir, "stage_bonds.mrc")
    mrc_fin = os.path.join(out_dir, "stage_final.mrc")

    write_mrc_time = time.time()
    write_mrc_with_geometry(ca_pts, n_pts, c_pts, args.label_mrc, mrc_thr)
    write_mrc_with_geometry(ca_nms, n_nms, c_nms, args.label_mrc, mrc_nms)
    write_mrc_with_geometry(ca_b, n_b, c_b, args.label_mrc, mrc_bnd)
    write_mrc_with_geometry(ca_f, n_f, c_f, args.label_mrc, mrc_fin)
    print(f"write_mrc_time: {(time.time() - write_mrc_time) :.2f} 秒")

    # 8) 計算 IoU（每階段）
    at_cm_time = time.time()
    at_cm, at_per_class, at_macro_f1, at_macro_iou, at_ap_per_class, at_map = report_iou_from_files("After Thresholding", args.label_mrc, mrc_thr, ca_prob=ca_pts, n_prob=n_pts, c_prob=c_pts, out_path=out_dir, R_sphere=args.r_sphere)
    print(f"af_thr_iou: {(time.time() - at_cm_time) :.2f} 秒")

    an_cm_time = time.time()
    an_cm, an_per_class, an_macro_f1, an_macro_iou, an_ap_per_class, an_map = report_iou_from_files("After NMS", args.label_mrc, mrc_nms, ca_prob=ca_nms, n_prob=n_nms, c_prob=c_nms, out_path=out_dir, R_sphere=args.r_sphere)
    print(f"nms_iou: {(time.time() - an_cm_time) :.2f} 秒")
    
    au_cm_time = time.time()
    au_cm, au_per_class, au_macro_f1, au_macro_iou, au_ap_per_class, au_map = report_iou_from_files("After Bonds Match", args.label_mrc, mrc_bnd, ca_prob=ca_b, n_prob=n_b, c_prob=c_b, out_path=out_dir, R_sphere=args.r_sphere)
    print(f"bond_iou: {(time.time() - au_cm_time) :.2f} 秒")

    af_cm_time = time.time()
    af_cm, af_per_class, af_macro_f1, af_macro_iou, af_ap_per_class, af_map = report_iou_from_files("Final", args.label_mrc, mrc_fin, ca_prob=ca_f, n_prob=n_f, c_prob=c_f, out_path=out_dir, R_sphere=args.r_sphere)
    print(f"final_iou: {(time.time() - af_cm_time) :.2f} 秒")
    
    trail_time = time.time()
    trial_record = {
        "Date" : str(date.today()),
        "sphere_radius":  args.r_sphere,
        "coverage_factor": coverage_factor,
        "ca_multiplier": ca_mult,
        "n_multiplier": n_mult,
        "c_multiplier": c_mult,
        "nms":{
            "org_nms_radius": nms_radius,
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
    print(f"write_tail_time: {(time.time() - trail_time) :.2f} 秒")
    return np.round(af_macro_iou, 4), trial_record, rows_all

#############################################################################################

def optuna_search(args, n_trials: int = 100, storage: Optional[str] = None,verbose: int = 1):
    """
    使用 Optuna 做超參數搜尋；支援 SQLite 持久化（異常終止可續跑）。
    - storage: SQLite 連線字串，例如 'sqlite:///optuna.db'
               若為 None，建議在 main() 先決定好並傳入。
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # ---- 搜尋空間（可依需要調整）----
        coverage_factor = trial.suggest_float("coverage_factor", 1.00, 2.00, step=0.1) 
        ca_mult         = trial.suggest_float("ca_multiplier",   0.80, 1.10, step=0.1)
        n_mult          = trial.suggest_float("n_multiplier",    0.80, 1.10, step=0.1)
        c_mult          = trial.suggest_float("c_multiplier",    0.80, 1.10, step=0.1)
        nms_radius      = trial.suggest_float("nms_radius",      1.10, 1.90, step=0.1)

        # angle 在外部進行匹配，避免重複執行tol、w
        use_angle_param = trial.suggest_categorical("use_angle",[False, True])
        # use_angle_param=False
        
        if use_angle_param is True:
            angle_tol_param = trial.suggest_float("angle_tol", 20.0, 30.0, step=1)
            w_angle_param = trial.suggest_float("w_angle", 0.7, 1.3, step=0.1)
        else: # 如果不用角度，賦予固定的、無意義的值 (例如 Params 類別的預設值)
            angle_tol_param = 0  
            w_angle_param = 0  

        # Bond matching 參數（僅限 Params 定義的）
        bm_params = Params(
            r_ca_c_max=trial.suggest_float("r_ca_c_max", 2.0, 5.0, step=0.5),
            r_ca_n_max=trial.suggest_float("r_ca_n_max", 2.0, 5.0, step=0.5),
            r_ca_c_low = 1.0 ,
            r_ca_n_low = 1.0 ,
            distance_power=trial.suggest_float("distance_power", 1.0, 1.5, step=0.1),
            w_dist_c=trial.suggest_float("w_dist_c", 0.7, 1.3, step=0.1),
            w_dist_n=trial.suggest_float("w_dist_n", 0.7, 1.3, step=0.1),

            angle_target=110.0,  # 固定不動
            angle_tol=angle_tol_param,
            w_angle=w_angle_param,
            
            bondlen_window=trial.suggest_categorical("bondlen_window", [False, True]), # [False, True]
            # bondlen_window = False ,  # 固定 False
            ca_c_len_lo=1.3, ca_c_len_hi=1.9,
            ca_n_len_lo=1.2, ca_n_len_hi=1.8,
            exclusive_match=True,  # 固定 True
        )

        trail_time = time.time()
        final_avg_iou, record, _rows = run_pipeline_with_params(
            args,
            coverage_factor=coverage_factor,
            ca_mult=ca_mult,
            n_mult=n_mult,
            c_mult=c_mult,
            nms_radius=nms_radius,
            bm_params=bm_params,
            tag_prefix="optuna")
        
        trial.set_user_attr("record", record)
        
        print(f"run_one_trail_time: {(time.time() - trail_time) :.2f} 秒")

        return final_avg_iou if not np.isnan(final_avg_iou) else -1e9

    # ---- 建立 study：使用 SQLite 持久化，可異常中斷後續跑 ----
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)

    # === 根據 verbose 控制 Optuna 輸出與進度條 ===
    if verbose <= 0:
        optuna.logging.set_verbosity(optuna.logging.ERROR)  # 幾乎不印
        show_bar = False
    elif verbose == 1:
        optuna.logging.set_verbosity(optuna.logging.INFO)   # 正常印 trial 訊息
        show_bar = True
    else:  # verbose >= 2
        optuna.logging.set_verbosity(optuna.logging.DEBUG)  # 超級囉嗦模式
        show_bar = True

    if storage:
        # load_if_exists=True 可讓你重複執行時接續既有 study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
            study_name="bondmatch_pipeline"
        )
    else:
        # 沒給 storage = 記憶體模式（不持久化，不可續跑）
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_bar)
    
    # ---- 輸出最佳紀錄 ----
    best_trial = study.best_trial
    best_record = best_trial.user_attrs.get("record", {})
    best_json = os.path.join(args.output_dir, args.best_json_name)
    with open(best_json, "w") as f:
        json.dump(best_record, f, indent=2)

    print(f"[OPTUNA] BEST avg IoU = {best_record.get('final_avg_iou','')}")
    print(f"[OPTUNA] BEST out_dir = {best_record.get('out_dir','')}")
    print(f"[OPTUNA] BEST final MRC = {best_record.get('final_mrc','')}")
    print(f"[OPTUNA] best record → {best_json}")

    # （可選）清理非最佳的 optuna_* 資料夾
    keep_name = os.path.basename(best_record.get('out_dir', ''))
    if keep_name:
        for folder in os.listdir(args.output_dir):
            full_path = os.path.join(args.output_dir, folder)
            if os.path.isdir(full_path) and folder.startswith("optuna_") and folder != keep_name:
                shutil.rmtree(full_path)

    return study

#############################################################################################

def build_parser():
    p = argparse.ArgumentParser(description="原子聚類優化")

    # 設定 version & emd_id
    p.add_argument("--version", required=True, help="Cryo2Struct 版本（預設：Cryo2Struct）")
    p.add_argument("--emd_id", required=True, help="EMD ID（如 12465）")
    
    # 可選；用 version+emd_id 自動推
    p.add_argument("--prob_file", default=None, help="概率文件（不給則自動依 version/emd_id 推導）")
    p.add_argument("--reference_map", default=None, help="參考MRC文件")
    p.add_argument("--fasta", default=None, help="FASTA文件")
    p.add_argument("--label_mrc", help="對照用的 label MRC（1=CA,2=N,3=C）")

    p.add_argument("--output_dir", default=None, help="中介 MRC 輸出資料夾")
    p.add_argument("--output", default=None, help="輸出MRC文件")
    p.add_argument("--stage_prefix", default=None, help="階段檔名前綴（預設沿用 --output 前綴）")
    p.add_argument("--metrics_csv", default=None, help="輸出 IoU/Precision/Recall 的 CSV")
    p.add_argument("--best_json_name", default=None, help="輸出最佳參數的 json 檔名")

    # 其他參數
    p.add_argument("--nms_radius", type=float, default=0.9, help="NMS 半徑")
    p.add_argument("--coverage_factor", type=float, default=None,help="Coverage factor（未指定時會 sweep）")
    p.add_argument("--r_sphere", type=float, default=0.0, help="sphere radius for IoU calculation")
    p.add_argument("--ca_txt", help="CA質心輸出文件")
    p.add_argument("--n_txt", help="N質心輸出文件")
    p.add_argument("--c_txt", help="C質心輸出文件")
    
    # 啟用網格搜尋
    p.add_argument("--optuna", action="store_true", help="使用 Optuna 進行超參數搜尋")
    p.add_argument("--optuna-trials", type=int, default=100, help="Optuna 搜尋的試次數")
    p.add_argument("--optuna-storage", type=str, default="", help="Optuna 儲存，例如 sqlite:///optuna.db（空字串=不用）；留空=自動設定在 <output_dir>/optuna_study.db")
    
    # 最佳條件
    p.add_argument("--objective",choices=["r_sphere", "count"],default="r_sphere",
    help="選擇最佳條件：r_sphere=Final avg IoU 最高；count=Final CA 最接近FASTA數量（預設：r_sphere）")
    
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 根據 version + emd_id 自動補缺少的參數
    if args.prob_file is None:
        base = f"/media/ray-suen/TRANSCEND1/huei/{args.version}/all_output/org_output/input/{args.emd_id}" # 檔案位置
        args.prob_file = os.path.join(base, f"{args.emd_id}_probabilities_atom.txt") # 檔名
    
    if args.reference_map is None:
        base = f"/media/ray-suen/TRANSCEND1/huei/{args.version}/all_output/org_output/input/{args.emd_id}" 
        args.reference_map = os.path.join(base, "emd_normalized_map.mrc")

    if args.output is None:
        if args.version == "Cryo2Struct":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}" 
                args.output = os.path.join(base, f"{args.emd_id}_v1_optuna_annotated.mrc")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}" 
                args.output = os.path.join(base, f"{args.emd_id}_v1_annotated.mrc")

        if args.version == "Cryo2Struct2":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}" 
                args.output = os.path.join(base, f"{args.emd_id}_v2_optuna_annotated.mrc")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}" 
                args.output = os.path.join(base, f"{args.emd_id}_v2_annotated.mrc")

    if args.fasta is None:
        fasta_dir = f"/media/ray-suen/TRANSCEND1/huei/org_fasta_and_label_map/{args.emd_id}"
        fasta_candidates = sorted(glob.glob(os.path.join(fasta_dir, "*.fasta")))
        if not fasta_candidates:
            sys.exit(f"錯誤: 在 {fasta_dir} 沒有找到 fasta 檔，請用 --fasta 指定")
        args.fasta = fasta_candidates[0] # 如果有多個，就取第一個
        print(f"[info] 自動使用 fasta: {args.fasta}")

    if args.label_mrc is None:
        base = f"/media/ray-suen/TRANSCEND1/huei/org_fasta_and_label_map/{args.emd_id}" 
        args.label_mrc = os.path.join(base, "round_off_atom_emd_normalized_map.mrc")

    if args.output_dir is None:
        if args.version == "Cryo2Struct":
            if args.optuna == True:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}/v1_optuna/point_radius_4.2"
            else:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}/v1/point_radius_4.2" 
        if args.version == "Cryo2Struct2":
            if args.optuna == True:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}/v2_optuna/point_radius_4.2"
            else:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}/v2/point_radius_4.2" 

    if args.stage_prefix is None:
        args.stage_prefix = f"{args.emd_id}"

    if args.metrics_csv is None:
        if args.version == "Cryo2Struct":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/one_sample/{args.emd_id}/v1_optuna" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_optuna_metrics.csv")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/new_way/org_flow(sphere&point)/{args.emd_id}/v1" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_round_off_metrics.csv")
        if args.version == "Cryo2Struct2":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/new_way/org_flow(sphere&point)/{args.emd_id}/v2_optuna" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_optuna_metrics.csv")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/new_way/org_flow(sphere&point)/{args.emd_id}/v2" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_round_off_metrics.csv")

    # 建立總輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 列印當前參數
    print("#"*20, "Running with below configuration","#"*20)
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(k, "=", v)
    
    if getattr(args, "optuna", False):
        # 若沒指定 storage，就預設放到 <output_dir>/optuna_study.db
        storage = args.optuna_storage.strip()
        if not storage:
            os.makedirs(args.output_dir, exist_ok=True)
            storage = f"sqlite:///{os.path.join(args.output_dir, 'optuna_study.db')}"
            print(f"[OPTUNA] 使用預設 SQLite 儲存：{storage}")
        else:
            print(f"[OPTUNA] 使用自訂 SQLite 儲存：{storage}")

        # trials 預設從參數來
        n_trials = int(getattr(args, "optuna_trials", 100))

        # 跑搜尋（你原本的函式名）
        args.best_json_name = f"{date.today()}_point_cf2_{args.emd_id}_{n_trials}_r_sphere_{args.r_sphere}_optuna_best.json"
        # args.best_json_name = args.best_json_name
        optuna_search(args, n_trials=n_trials, storage=storage)
        return

if __name__ == "__main__":
    print("Date:",date.today())
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"總執行時間: {elapsed_time:.2f} 秒")

start_time = time.time()
elapsed_time = time.time() - start_time
print(f"總執行時間: {(time.time() - start_time) :.2f} 秒")