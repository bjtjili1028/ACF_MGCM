# Evaluation Metrics and Tool Usage

本文件說明本研究中使用之評估指標與相關工具，包含原子偵測層級之 **IoU / F1-score**，以及整體結構層級之 **TM-score / RMSD**。此外，也說明如何將 PDB 或 CIF 結構轉換為 MRC label map，以供後續評估流程使用。

---

## 1. 評估流程概述

本研究主要使用兩類評估方式：

1. **原子偵測層級評估**

   * 使用 IoU 與 F1-score 評估預測原子點與參考標註之間的重疊與匹配程度。
   * 相關程式：`cal_iou.py`

2. **整體結構層級評估**

   * 使用 USalign 計算 TM-score 與 RMSD。
   * 使用 `phenix.superpose_pdbs` 進行 PDB 結構疊合與 RMSD 評估。
   * 相關工具：`USalign`、`phenix.superpose_pdbs`

---

## 2. PDB / CIF 轉換為 MRC Label Map

在進行 IoU / F1-score 評估前，需要先將參考結構轉換為 MRC label map。
本專案提供兩個轉換程式，分別支援 PDB 與 CIF 格式。

---

### 2.1 PDB to MRC：`get_atoms_label.py`

`get_atoms_label.py` 用於將 PDB 結構資料轉換為 MRC label map，主要適用於 Cryo2Struct V1 與 Cryo2Struct V2 的結構資料。

#### 使用前設定

執行前請先開啟檔案：

```bash
ACF_MGCM/eval/get_atoms_label.py
```

並於程式第 **151 至 154 行** 選擇或重新輸入欲處理的檔案名稱與相關路徑設定。

#### 執行方式

```bash
python3 ACF_MGCM/eval/get_atoms_label.py [存放 PDB 結構資料夾]
```

#### 範例

```bash
python3 ACF_MGCM/eval/get_atoms_label.py /path/to/pdb_folder
```

---

### 2.2 CIF to MRC：`get_atoms_label_cif.py`

`get_atoms_label_cif.py` 用於將 CIF 結構資料轉換為 MRC label map，主要適用於 CryoAtom。

#### 執行方式

```bash
python3 /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/eval/get_atoms_label_cif.py [存放 CIF 結構資料夾]
```

#### 範例

```bash
python3 /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/eval/get_atoms_label_cif.py /path/to/cif_folder
```

---

## 3. IoU / F1-score 評估

### 3.1 程式說明：`cal_iou.py`

`cal_iou.py` 用於計算預測結果與參考 label map 之間的 IoU 與 F1-score。
使用前請先進入程式內部，確認並修改對應的資料路徑、模型版本與輸出位置。

#### 需要確認的項目

執行前建議檢查以下設定：

* EMD ID 是否正確
* 預測結果路徑是否正確
* 參考 label map 路徑是否正確
* 模型版本設定是否正確
* 輸出結果儲存路徑是否正確

#### 執行方式

```bash
python3 -u ACF_MGCM/eval/cal_iou.py \
    --emd_id 0690 \
    --R_distance 4.2
```

#### 參數說明

| 參數             | 說明                   |
| -------------- | -------------------- |
| `--emd_id`     | 欲評估的 EMD ID          |
| `--R_distance` | 原子匹配或 IoU 計算時使用的距離半徑 |

#### 範例

```bash
python3 -u ACF_MGCM/eval/cal_iou.py \
    --emd_id 0690 \
    --R_distance 4.2
```

---

## 4. TM-score / RMSD 評估

整體結構層級的幾何相似度主要透過 USalign 與 Phenix 進行評估。

---

### 4.1 USalign 

USalign 可用於比較預測結構與參考結構之間的整體拓撲相似度，並輸出 TM-score 與 RMSD 等指標。

#### 執行方式

```bash
ACF_MGCM/eval/USalign [預測結果路徑] [參考答案路徑] -mm 1 -ter 1
```
* 若不能執行請檢查執行權限，或是重新透過cpp進行安裝。

#### 範例

```bash
ACF_MGCM/eval/USalign prediction.pdb ground_truth.pdb -mm 1 -ter 1
```

#### 輸出指標

常用輸出包含：

| 指標             | 說明               |
| -------------- | ---------------- |
| TM-score       | 衡量兩個蛋白質結構整體拓撲相似度 |
| RMSD           | 疊合後對應原子之均方根偏差    |
| Aligned length | 成功比對的殘基或原子數量     |

---

### 4.2 Phenix：`phenix.superpose_pdbs`

`phenix.superpose_pdbs` 可用於對兩個 PDB 結構進行疊合，並輸出 RMSD 等結構差異資訊。

#### 安裝需求

使用前請先安裝 Phenix，並確認 `phenix.superpose_pdbs` 指令可以正常執行。

可先使用以下指令確認：

```bash
phenix.superpose_pdbs --help
```

#### 執行方式

```bash
phenix.superpose_pdbs ground_truth.pdb prediction.pdb
```

#### 範例

```bash
phenix.superpose_pdbs reference.pdb predicted.pdb
```

---

## 5. 建議執行順序

建議依照以下流程進行完整評估：

### Step 1：準備參考結構

若參考資料為 PDB：

```bash
python3 ACF_MGCM/eval/get_atoms_label.py [存放 PDB 結構資料夾]
```

若參考資料為 CIF：

```bash
python3 /media/ray-suen/TRANSCEND1/huei/ACF_MGCM/eval/get_atoms_label_cif.py [存放 CIF 結構資料夾]
```

---

### Step 2：計算 IoU / F1-score

```bash
python3 -u ACF_MGCM/eval/cal_iou.py \
    --emd_id 0690 \
    --R_distance 4.2
```

---

### Step 3：使用 USalign 計算 TM-score / RMSD

```bash
ACF_MGCM/eval/USalign prediction.pdb ground_truth.pdb -mm 1 -ter 1
```

---

### Step 4：使用 Phenix 進行結構疊合

```bash
phenix.superpose_pdbs ground_truth.pdb prediction.pdb
```

---

## 6. 注意事項

1. 執行 `get_atoms_label.py` 前，請確認程式第 151 至 154 行的檔案名稱與路徑設定是否正確。
2. 執行 `cal_iou.py` 前，請確認程式內部的資料路徑與模型版本設定。
3. 若使用 CryoAtom 的 CIF 結構資料，請使用 `get_atoms_label_cif.py` 進行轉換。
4. USalign 的輸入順序建議為：

```bash
USalign [預測結果路徑] [參考答案路徑] -mm 1 -ter 1
```

5. `phenix.superpose_pdbs` 需要事先安裝 Phenix，並確認該指令已加入系統環境變數。
6. 若路徑中包含空白字元，請使用引號包住路徑，例如：

```bash
ACF_MGCM/eval/USalign "path/to/prediction.pdb" "path/to/ground truth.pdb" -mm 1 -ter 1
```

---

## 7. 相關檔案位置

| 檔案 / 工具                                | 功能                        |
| -------------------------------------- | ------------------------- |
| `ACF_MGCM/eval/get_atoms_label.py`     | 將 PDB 結構轉換為 MRC label map |
| `ACF_MGCM/eval/get_atoms_label_cif.py` | 將 CIF 結構轉換為 MRC label map |
| `ACF_MGCM/eval/cal_iou.py`             | 計算 IoU 與 F1-score         |
| `ACF_MGCM/eval/USalign`                | 計算 TM-score 與 RMSD        |
| `phenix.superpose_pdbs`                | 進行 PDB 結構疊合與 RMSD 評估      |

---

## 8. Example Commands

以下為完整評估流程範例：

```bash
# 1. Convert PDB to MRC label map
python3 ACF_MGCM/eval/get_atoms_label.py /path/to/pdb_folder

# 2. Calculate IoU / F1-score
python3 -u ACF_MGCM/eval/cal_iou.py \
    --emd_id 0690 \
    --R_distance 4.2

# 3. Calculate TM-score / RMSD by USalign
ACF_MGCM/eval/USalign prediction.pdb ground_truth.pdb -mm 1 -ter 1

# 4. Superpose structures by Phenix
phenix.superpose_pdbs ground_truth.pdb prediction.pdb
```
