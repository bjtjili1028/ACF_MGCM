# Adaptive candidate filtering (ACF) & Multi-stage Geometric Constraint Matching（MGCM）


![ACF & MGCM_overview](./img/ACF_MGCM_flow.png)

## 環境設定
若要在本機安裝 ACF & MGCM ，請依照下列步驟操作。

Clone this repository and `cd` into it
```
git clone https://github.com/bjtjili1028/ACF_MGCM.git
cd ./ACF_MGCM
```

使用 Anaconda 設定環境。以下範例展示如何設定 conda 環境來運行程式碼。使用以下命令透過該``environment.yml``檔案建立 conda 環境。
```
conda env create -f environment.yml
conda activate ACF_MGCM
```

## 使用 ACF & MGCM 進行聚類和原子匹配

1. <ins>**輸入低溫電子密度圖和序列**</ins>：首先，您需要準備自己的資料或使用我們提供的範例資料。目錄結構應如下：

```text 
ACF_MGCM
|── input
    │── ca_atom or three_atom
        |-- 7quc.fasta
        │-- 14147_probabilities_atom.txt
        │-- emd_normalized_map.mrc
        │-- round_off_atom_emd_normalized_map.mrc
```
``7quc.fasta`` 是 EMD ID 為 34610 的序列檔。 

``14147_probabilities_atom.txt`` 是 Cryo2Struct 模型預測階段輸出的檔案。

``emd_normalized_map.mrc`` 和``round_off_atom_emd_normalized_map.mrc`` 是透過 Cryo2Struct Data 所產生的檔案。

2. <ins>**修改資料路徑和超參數**</ins>：請至 ``config/arguments_point_ca.yml``中，修改資料路徑和欲使用的超參數。

3. <ins>**訓練**</ins>：請執行

```
python3 main_point_ca.py 
```
程式就會依據您輸入的資料路徑、超參數及指定的輸出資料夾產生出下列對應的``json``檔，便於觀察結果。


上述是以單原子為例進行說明，多原子的部分就是上述將上述檔名的``_ca``移除。
## 訓練超參數

請參閱 ``train/Training Hyperparameters.md``

## 評估指標使用

請參閱 ``eval/Evaluation.md``

## 關聯資料庫
資料庫的設置和下載請參考各資料庫的 READMD 。
1. [Cryo2Struct GitHub](https://github.com/bjtjili1028/Cryo2struct.git)
    - 模型輸出檔案請使用：``<EMD_ID>_probabilities_atom.txt``
2. [Cryo2Struct2 GitHub](https://github.com/bjtjili1028/Cryo2struct2.git)
    - 模型輸出檔案請使用：``<EMD_ID>_probabilities_atom.txt``
3. [CryoAtom GitHub](https://github.com/bjtjili1028/CryoAtom.git)
    - 模型輸出檔案請使用：``CA_before_clustering.txt``
4. [MICA GitHub](https://github.com/bjtjili1028/MICA.git)
    - 模型輸出檔案請使用：``<EMD_ID>_CA_before_clustering.txt``