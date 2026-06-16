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
from utils.atom_pick_ca import determine_target_count, adaptive_threshold_analysis, parse_probabilities, nms_kdtree_adaptive,resolve_cross_class_overlaps_keep_maxprob
from utils.point_cal_iou import write_mrc_with_geometry_CA, report_iou_from_files_CA

# 超參數主要搜索流程
def run_pipeline_with_params(args,
                             coverage_factor: float,
                             ca_mult: float,
                             nms_radius: float,
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
    # ca_t, n_t, c_t = adaptive_threshold_analysis(args.prob_file, target_coverage)
    ca_t = adaptive_threshold_analysis(args.prob_file, target_coverage)
    def clamp01(x): return max(0.0, min(1.0, float(x)))
    
    ca_t = clamp01(ca_t * ca_mult)

    print(f"[GRID] thresholds  CA={ca_t:.4f} (x{ca_mult})")
    print(f"計算自適應閾值，乘上倍率: {(time.time() - adp_start_time) :.2f} 秒")

    # 3) 解析概率點
    use_adp_start_time = time.time()
    ca_pts = parse_probabilities(args.prob_file, ca_t)
    count_thr = {1: len(ca_pts)}
    n_ca = len(ca_pts)
    if n_ca == 0 :     # 可以設更嚴格的下界
        raise optuna.exceptions.TrialPruned("Empty candidates after thresholding.")
    print(f"使用自適應閾值: {(time.time() - use_adp_start_time) :.2f} 秒")

    # 4) NMS
    nms_start_time = time.time()
    ca_nms,ca_final_r = nms_kdtree_adaptive(ca_pts,nms_radius, max_points=target_coverage)
    count_nms = {1: len(ca_nms)}
    print(f"nms: {(time.time() - nms_start_time) :.2f} 秒")

    # 5) 寫出各階段 MRC（只用於 IoU；檔名加上 tag)
    tag = f"{tag_prefix}_cf{coverage_factor:.2f}_nm{nms_radius:.2f}_mul{ca_mult:.2f}"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    mrc_thr = os.path.join(out_dir, "stage_threshold.mrc")
    mrc_nms = os.path.join(out_dir, "stage_nms.mrc")
    mrc_fin = os.path.join(out_dir, "stage_final.mrc")

    print("write_mrc_with_geometry_CA")
    write_mrc_time = time.time()
    write_mrc_with_geometry_CA(ca_pts, args.label_mrc, mrc_thr)
    write_mrc_with_geometry_CA(ca_nms, args.label_mrc, mrc_nms)
    print(f"write_mrc_time: {(time.time() - write_mrc_time) :.2f} 秒")

    # 6) 計算 IoU（每階段）
    at_cm_time = time.time()
    at_cm, at_per_class, at_macro_f1, at_macro_iou, at_ap_per_class, at_map = report_iou_from_files_CA("After Thresholding", args.label_mrc, mrc_thr, ca_prob=ca_pts, out_path=out_dir, R_sphere=args.r_sphere)
    print(f"af_thr_iou: {(time.time() - at_cm_time) :.2f} 秒")

    an_cm_time = time.time()
    an_cm, an_per_class, an_macro_f1, an_macro_iou, an_ap_per_class, an_map = report_iou_from_files_CA("After NMS", args.label_mrc, mrc_nms, ca_prob=ca_nms, out_path=out_dir, R_sphere=args.r_sphere)
    print(f"nms_iou: {(time.time() - an_cm_time) :.2f} 秒")
    
    trail_time = time.time()
    trial_record = {
        "Date" : str(date.today()),
        "sphere_radius":  args.r_sphere,
        "coverage_factor": coverage_factor,
        "ca_multiplier": ca_mult,
        "nms":{
            "org_nms_radius": nms_radius,
            "ca_final_nms_r": ca_final_r,
        },
        "counts": {
            "After Thresholding": count_thr,
            "After NMS": count_nms,
        },
        "CA_atom_iou_metrics": {
            "After Thresholding": at_per_class["CA"],
            "After NMS": an_per_class["CA"],
        },
        "confusion_matrix": {
            "After Thresholding": at_cm,
            "After NMS": an_cm,
        },
        "AP_per_class": {
            "After Thresholding": at_ap_per_class,
            "After NMS": an_ap_per_class,
        },
        "avg_iou_metrics": {
            "After Thresholding": at_macro_iou,
            "After NMS": an_macro_iou,
        },
        "avg_f1_metrics": {
            "After Thresholding": at_macro_f1,
            "After NMS": an_macro_f1,
        },
        "avg_AP_metrics": {
            "After Thresholding": at_map,
            "After NMS": an_map,
        },
        "final_avg_iou": np.round(an_macro_iou, 4),
        "final_f1": np.round(an_macro_f1, 4),
        "final_map": np.round(an_map, 4),
        "out_dir": out_dir,
        "final_mrc": mrc_fin
    }
    print(f"write_tail_time: {(time.time() - trail_time) :.2f} 秒")
    return np.round(an_macro_iou, 4), trial_record, rows_all

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
        nms_radius      = trial.suggest_float("nms_radius",      1.10, 1.90, step=0.1)


        trail_time = time.time()
        final_avg_iou, record, _rows = run_pipeline_with_params(
            args,
            coverage_factor=coverage_factor,
            ca_mult=ca_mult,
            nms_radius=nms_radius,
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
    try:
        best_trial = study.best_trial
    except ValueError:
        print("[OPTUNA] 在資料庫中沒有找到任何成功的 trial (可能是所有 trial 都因為某個錯誤 failed 或是被 pruned 了)。請檢查前面的日誌獲取錯誤原因。")
        return study
        
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
        if args.version == "Cryo2Struct":
            base = f"/media/ray-suen/TRANSCEND1/huei/Cryo2Struct/all_output/org_output/input/{args.emd_id}" 
            args.prob_file = os.path.join(base, f"{args.emd_id}_probabilities_atom.txt")

        if args.version == "Cryo2Struct2":
            base = f"/media/ray-suen/TRANSCEND1/huei/Cryo2Struct2/all_output/test_lamda_orgput/lamda_15/input/{args.emd_id}" 
            args.prob_file = os.path.join(base, f"{args.emd_id}_probabilities_atom.txt")

        if args.version == "MICA":
            base = f"/media/ray-suen/TRANSCEND1/huei/MICA/output/results/{args.emd_id}/final_output" # 檔案位置
            args.prob_file = os.path.join(base, f"{args.emd_id}_CA_before_clustering.txt") # 檔名

        if args.version == "Cryo_Atom":
            base = f"/media/ray-suen/TRANSCEND1/huei/CryoAtom/out/{args.emd_id}/see_alpha_output" # 檔案位置
            args.prob_file = os.path.join(base, f"CA_before_clustering.txt") # 檔名
    
    if args.reference_map is None:
        base = f"/media/ray-suen/TRANSCEND1/huei/org_fasta_and_label_map/{args.emd_id}" 
        args.reference_map = os.path.join(base, "emd_normalized_map.mrc")

    if args.output is None:
        if args.version == "Cryo2Struct":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/one_sample/{args.emd_id}" 
                args.output = os.path.join(base, f"{args.emd_id}_v1_optuna_annotated.mrc")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/one_sample/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_v1_annotated.mrc")

        if args.version == "Cryo2Struct2":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct2_optuna_ca/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_v2_optuna_annotated.mrc")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct2_optuna_ca/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_v2_annotated.mrc")

        if args.version == "MICA":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_MICA_optuna_annotated.mrc")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_MICA_annotated.mrc")
                # /media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/0689

        if args.version == "Cryo_Atom":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_Cryo_Atom_optuna_annotated.mrc")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/{args.emd_id}/owner_before_clustering_train" 
                args.output = os.path.join(base, f"{args.emd_id}_Cryo_Atom_annotated.mrc")
                # /media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/0689/owner_before_clustering_train

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
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/one_sample/{args.emd_id}"
            else:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct_optuna_ca/{args.emd_id}/owner_before_clustering_train" 
        if args.version == "Cryo2Struct2":
            if args.optuna == True:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct2_optuna_ca/{args.emd_id}/owner_before_clustering_train"
            else:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct2_optuna_ca/{args.emd_id}/owner_before_clustering_train" 
        if args.version == "MICA":
            if args.optuna == True:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/{args.emd_id}/owner_before_clustering_train"
            else:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/{args.emd_id}/owner_before_clustering_train" 
        if args.version == "Cryo_Atom":
            if args.optuna == True:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/{args.emd_id}/owner_before_clustering_train"
            else:
                args.output_dir = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/{args.emd_id}/owner_before_clustering_train" 
            

    if args.stage_prefix is None:
        args.stage_prefix = f"{args.emd_id}"

    if args.metrics_csv is None:
        if args.version == "Cryo2Struct":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct_optuna_ca/{args.emd_id}/owner_before_clustering_train" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_optuna_metrics.csv")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo2struct_optuna_ca/{args.emd_id}/owner_before_clustering_train" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_round_off_metrics.csv")
        if args.version == "Cryo2Struct2":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/new_way/org_flow(sphere&point)/{args.emd_id}/v2_optuna" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_optuna_metrics.csv")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/new_way/org_flow(sphere&point)/{args.emd_id}/v2" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_round_off_metrics.csv")
        if args.version == "MICA":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/{args.emd_id}" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_optuna_metrics.csv")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/MICA_optuna/{args.emd_id}" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_round_off_metrics.csv")
        if args.version == "Cryo_Atom":
            if args.optuna == True:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/{args.emd_id}/owner_before_clustering_train" 
                args.metrics_csv = os.path.join(base, f"{args.emd_id}_optuna_metrics.csv")
            else:
                base = f"/media/ray-suen/TRANSCEND1/huei/v1_v2_final/Cryo_Atom_out/{args.emd_id}/owner_before_clustering_train" 
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