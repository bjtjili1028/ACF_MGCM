## Training Hyperparameters

本節說明本專案用於搜尋自適應候選原子篩選機制（Adaptive Candidate Filtering, ACF）和多階段幾何約束匹配（Multi-stage Geometric Constraint Matching（MGCM））超參數之訓練程式。訓練流程主要透過 Optuna 進行參數搜尋，並依據使用樣本數量區分為單一樣本訓練（ACF-S）與多樣本共同訓練（ACF-G）。

---

### 1. 使用前注意事項

執行訓練程式前，請先進入專案根目錄，並確認目前工作路徑位於：

```bash
/media/ray-suen/TRANSCEND1/huei/ACF_MGCM
```

接著進入 `train` 資料夾檢查各訓練程式內部設定：

```bash
cd /media/ray-suen/TRANSCEND1/huei/ACF_MGCM
```

請特別注意：每一個訓練程式在執行前，都需要重新確認並修改程式內部的資料路徑設定。輸入資料應為各模型在預測區塊輸出的原始候選點資料，亦即尚未經過聚類、NMS 或其他後處理流程的預測結果。

執行前建議確認以下項目：

| 項目           | 說明                             |
| ------------ | ------------------------------ |
| 預測結果路徑       | 各模型輸出的原始候選原子點資料                |
| label map 路徑 | 由 PDB / CIF 轉換後的 MRC label map |
| 輸出資料夾        | Optuna 搜尋結果、log 檔與最佳參數儲存位置     |
| 模型版本         | 例如 `Cryo2Struct`               |
| EMD ID       | 欲訓練或評估的樣本編號                    |
| `r_sphere`   | IoU / F1-score 計算時使用的匹配半徑      |

---

## 2. 單一樣本訓練：ACF-S

單一樣本訓練用於針對特定 EMD 樣本進行參數搜尋。此設定對應論文中的 **ACF-S**，即 sample-specific adaptive configuration。

---

### 2.1 `one_sample_optuna_search_point_ca.py`

此程式僅針對 **Cα 單原子模型** 的預測輸出進行訓練，適用於只評估或最佳化 Cα 原子定位結果的情境。

#### 功能說明

* 輸入：Cα 模型輸出的原始候選點資料
* 任務：搜尋單一樣本下的最佳 ACF 超參數
* 訓練方式：單一樣本訓練
* 對應設定：ACF-S

#### 執行範例

```bash
python3 -u train/one_sample_optuna_search_point_ca.py \
    --version Cryo2Struct \
    --emd_id 0689 \
    --r_sphere 4.2 \
    --optuna \
    --optuna-trials 10 \
    > "0689_ca_bond_notset_100_optuna_round_off.log" 2>&1 &
```

---

### 2.2 `one_sample_optuna_search_point.py`

此程式僅針對 **Cryo2Struct 三原子模型** 的預測輸出進行訓練，適用於同時處理 Cα、N、C 三類原子的候選點篩選與參數最佳化。

#### 功能說明

* 輸入：Cryo2Struct 三原子模型輸出的原始候選點資料
* 原子類別：Cα、N、C
* 任務：搜尋單一樣本下的最佳 ACF 超參數
* 訓練方式：單一樣本訓練
* 對應設定：ACF-S

#### 執行範例

```bash
python3 -u train/one_sample_optuna_search_point.py \
    --version Cryo2Struct \
    --emd_id 0689 \
    --r_sphere 4.2 \
    --optuna \
    --optuna-trials 10 \
    > "0689_three_atom_notset_100_optuna_round_off.log" 2>&1 &
```

---

## 3. 多樣本共同訓練：ACF-G

多樣本共同訓練用於同時使用多個 EMD 樣本進行 Optuna 參數搜尋。此設定對應論文中的 **ACF-G**，即 generalized adaptive configuration。其目的在於取得一組可跨樣本使用的泛化參數，而非只針對單一樣本最佳化。

---

### 3.1 `ten_sample_optuna_search_point_ca.py`

此程式僅針對 **Cα 單原子模型** 的預測輸出進行多樣本訓練。

#### 功能說明

* 輸入：多個樣本的 Cα 模型原始候選點資料
* 任務：搜尋跨樣本通用的 Cα ACF 超參數
* 訓練方式：多樣本共同訓練
* 對應設定：ACF-G
* 支援 Optuna 平行搜尋

#### 執行範例

```bash
python3 -u train/ten_sample_optuna_search_point_ca.py \
    --emd_list "0689" "0690" "13256" "21652" "21708" "24794" "25224" "25225" "32351" "33861" \
    --version Cryo2Struct \
    --optuna \
    --r_sphere 4.2 \
    --optuna-n-jobs 4 \
    --output_dir "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/ten_sample" \
    --optuna-storage "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/ten_sample/optuna_study.db" \
    --optuna-trials 10 \
    > /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/ten_sample/ten_sample_notset_100_optuna_round_off.log 2>&1 &
```

---

### 3.2 `ten_sample_optuna_search_point.py`

此程式僅針對 **Cryo2Struct 三原子模型** 的預測輸出進行多樣本訓練，適用於同時最佳化 Cα、N、C 三類原子的候選點篩選參數。

#### 功能說明

* 輸入：多個樣本的 Cryo2Struct 三原子模型原始候選點資料
* 原子類別：Cα、N、C
* 任務：搜尋跨樣本通用的三原子 ACF 超參數
* 訓練方式：多樣本共同訓練
* 對應設定：ACF-G
* 支援 Optuna 平行搜尋

#### 執行範例

```bash
python3 -u train/ten_sample_optuna_search_point.py \
    --emd_list "0689" "0690" "13256" "21652" "21708" "24794" "25224" "25225" "32351" "33861" \
    --version Cryo2Struct \
    --optuna \
    --r_sphere 4.2 \
    --optuna-n-jobs 4 \
    --output_dir "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/ten_sample" \
    --optuna-storage "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/ten_sample/optuna_study.db" \
    --optuna-trials 10 \
    > /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/ten_sample/ten_sample_notset_100_optuna_round_off.log 2>&1 &
```

---

## 4. 參數說明

| 參數                 | 說明                            |
| ------------------ | ----------------------------- |
| `--version`        | 指定模型版本，例如 `Cryo2Struct`       |
| `--emd_id`         | 單一樣本訓練時指定 EMD ID              |
| `--emd_list`       | 多樣本訓練時指定多個 EMD ID             |
| `--r_sphere`       | 計算 IoU / F1-score 時使用的匹配半徑    |
| `--optuna`         | 啟用 Optuna 超參數搜尋               |
| `--optuna-trials`  | 指定 Optuna 搜尋次數                |
| `--optuna-n-jobs`  | 指定 Optuna 平行執行數量              |
| `--output_dir`     | 指定訓練結果輸出資料夾                   |
| `--optuna-storage` | 指定 Optuna study database 儲存位置 |

---

## 5. 訓練程式比較

| 程式名稱                                   | 原子類型       | 樣本數量 | 對應設定  | 功能                 |
| -------------------------------------- | ---------- | ---- | ----- | ------------------ |
| `one_sample_optuna_search_point_ca.py` | Cα         | 單一樣本 | ACF-S | 搜尋單一樣本的 Cα ACF 參數  |
| `one_sample_optuna_search_point.py`    | Cα / N / C | 單一樣本 | ACF-S | 搜尋單一樣本的三原子 ACF 參數  |
| `ten_sample_optuna_search_point_ca.py` | Cα         | 多樣本  | ACF-G | 搜尋跨樣本通用的 Cα ACF 參數 |
| `ten_sample_optuna_search_point.py`    | Cα / N / C | 多樣本  | ACF-G | 搜尋跨樣本通用的三原子 ACF 參數 |

---

## 6. 輸出結果

訓練完成後，程式會依照 `--output_dir` 設定輸出相關結果。常見輸出內容包含：

| 輸出內容                  | 說明                         |
| --------------------- | -------------------------- |
| Optuna study database | 儲存每一次 trial 的參數與結果         |
| log 檔                 | 記錄訓練過程、每次 trial 的分數與錯誤訊息   |
| 最佳參數結果                | 儲存搜尋後的最佳超參數組合              |
| 評估結果                  | 儲存對應樣本的 IoU / F1-score 等指標 |

---

## 7. 注意事項

1. 執行前請務必確認每個訓練程式內部的資料路徑已修改完成。
2. 輸入資料應為模型原始預測輸出，不能是已經過聚類、NMS 或其他後處理後的結果。
3. 單一樣本訓練對應 ACF-S，多樣本共同訓練對應 ACF-G。
4. 若使用 `--optuna-n-jobs` 進行平行搜尋，請確認 CPU / GPU / 記憶體資源是否足夠。
5. 建議將 log 輸出至指定檔案，方便後續檢查訓練結果與錯誤訊息。
6. 若執行時出現 `ModuleNotFoundError: No module named 'utils'`，請確認是否從專案根目錄執行程式，或在訓練程式開頭加入專案根目錄至 `sys.path`。

---

## 8. Example Commands

### 單一樣本 Cα 訓練

```bash
python3 -u train/one_sample_optuna_search_point_ca.py \
    --version Cryo2Struct \
    --emd_id 0689 \
    --r_sphere 4.2 \
    --optuna \
    --optuna-trials 10 \
    > "0689_ca_bond_notset_100_optuna_round_off.log" 2>&1 &
```

### 單一樣本三原子訓練

```bash
python3 -u train/one_sample_optuna_search_point.py \
    --version Cryo2Struct \
    --emd_id 0689 \
    --r_sphere 4.2 \
    --optuna \
    --optuna-trials 10 \
    > "0689_three_atom_notset_100_optuna_round_off.log" 2>&1 &
```

### 多樣本 Cα 訓練

```bash
python3 -u train/ten_sample_optuna_search_point_ca.py \
    --emd_list "0689" "0690" "13256" "21652" "21708" "24794" "25224" "25225" "32351" "33861" \
    --version Cryo2Struct \
    --optuna \
    --r_sphere 4.2 \
    --optuna-n-jobs 4 \
    --output_dir "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/ten_sample" \
    --optuna-storage "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/ten_sample/optuna_study.db" \
    --optuna-trials 10 \
    > /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/ca_result/ten_sample/ten_sample_notset_100_optuna_round_off.log 2>&1 &
```

### 多樣本三原子訓練

```bash
python3 -u train/ten_sample_optuna_search_point.py \
    --emd_list "0689" "0690" "13256" "21652" "21708" "24794" "25224" "25225" "32351" "33861" \
    --version Cryo2Struct \
    --optuna \
    --r_sphere 4.2 \
    --optuna-n-jobs 4 \
    --output_dir "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/ten_sample" \
    --optuna-storage "/media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/ten_sample/optuna_study.db" \
    --optuna-trials 10 \
    > /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/output/train/three_atom/ten_sample/ten_sample_notset_100_optuna_round_off.log 2>&1 &
```
