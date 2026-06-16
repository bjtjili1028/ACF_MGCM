"""
Updated for CIF support
Labels CA to 1, N to 2 and C to 3 in an MRC map based on a CIF structure.
"""
import sys
import mrcfile
import math
import numpy as np
import os
# [修改] 引入 MMCIFParser 以支援 .cif 格式
from Bio.PDB import MMCIFParser

def get_index(cord, origin, voxel):
    return round((cord - origin) / voxel)

def label_mask(path, org_map, cif_file, outfilename):
    print(f"--> Processing Map: {org_map}")
    print(f"--> Using Structure: {cif_file}")
    
    count = 0
    org_map_path = os.path.join(path, org_map)
    cif_file_path = os.path.join(path, cif_file)
    
    # 檢查檔案是否存在
    if not os.path.exists(org_map_path) or not os.path.exists(cif_file_path):
        print(f"Error: File not found. \nMap: {org_map_path}\nCIF: {cif_file_path}")
        return

    # 讀取 MRC Header 資訊
    try:
        with mrcfile.open(org_map_path, mode='r') as mrc:
            x_origin = mrc.header.origin['x']
            y_origin = mrc.header.origin['y']
            z_origin = mrc.header.origin['z']
            x_voxel = mrc.voxel_size['x']
            y_voxel = mrc.voxel_size['y']
            z_voxel = mrc.voxel_size['z']
            data_shape = mrc.data.shape
    except Exception as e:
        print(f"Error reading MRC file: {e}")
        return

    # 初始化空的資料陣列
    data = np.zeros(data_shape, dtype=np.float32)

    # [修改] 使用 MMCIFParser 讀取 .cif 結構
    parser = MMCIFParser(QUIET=True) # QUIET=True 減少不必要的警告訊息
    try:
        # get_structure 第一個參數是 ID，可以隨意給
        struct = parser.get_structure("structure", cif_file_path)
    except Exception as e:
        print(f"Error parsing CIF file: {e}")
        return
    
    # 逐一檢查所有原子
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.get_coord()
                    
                    # 計算索引 (並確保在邊界內，防止 Index Out of Bounds)
                    iz = int(get_index(z, z_origin, z_voxel))
                    jy = int(get_index(y, y_origin, y_voxel))
                    kx = int(get_index(x, x_origin, x_voxel))
                    
                    # 邊界檢查
                    if (0 <= iz < data_shape[0] and 
                        0 <= jy < data_shape[1] and 
                        0 <= kx < data_shape[2]):
                        
                        atom_name = atom.get_name()
                        # [注意] CIF 檔有時原子名稱會有差異，Biopython 通常會處理好
                        if atom_name == "CA":
                            data[iz, jy, kx] = 1
                            count += 1
                        elif atom_name == "N":
                            data[iz, jy, kx] = 2
                        elif atom_name == "C":
                            data[iz, jy, kx] = 3
    
    # 儲存結果
    out_path = os.path.join(path, outfilename)
    print(f"--> Saving output to: {outfilename}")
    
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        # mrc.header.origin = np.array([x_origin, y_origin, z_origin])
        mrc.header.origin['x'] = x_origin
        mrc.header.origin['y'] = y_origin
        mrc.header.origin['z'] = z_origin
        # 複製原始 header 的一些重要資訊 (可選)
        # mrc.header.map = mrcfile.open(org_map_path, header_only=True).header.map
    
    print(f"Done. Number of CA atoms labeled: {count}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_folder_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    
    # 獲取資料夾列表 (排除非資料夾的項目)
    # 邏輯：假設 input_path 下面有很多子資料夾，每個資料夾內有一組 map 和 cif
    try:
        map_names = [fn for fn in os.listdir(input_path) 
                     if os.path.isdir(os.path.join(input_path, fn))]
    except FileNotFoundError:
        print(f"Error: The path '{input_path}' does not exist.")
        sys.exit(1)

    if not map_names:
        print(f"No subdirectories found in {input_path}. \nTrying to process files in the root directory directly...")
        # 如果沒有子資料夾，嘗試直接處理輸入路徑
        map_names = ["."] 

    print("########### Generating atoms label (CIF Supported) ##########")

    for folder_name in map_names:
        # 處理路徑
        if folder_name == ".":
            path = input_path
        else:
            path = os.path.join(input_path, folder_name)

        # 1. 尋找 MRC 檔案
        emd_files = [f for f in os.listdir(path) if f.endswith(".mrc") and "final_atom" not in f]
        if not emd_files:
            if folder_name != ".": print(f"Skipping {folder_name}: No .mrc file found.")
            continue
        
        target_map = emd_files[0] # 取第一個找到的 map

        # 2. 尋找 CIF 檔案
        # 邏輯：先找 .cif 結尾的檔案
        all_cifs = [f for f in os.listdir(path) if f.endswith(".cif")]
        
        target_cif = None
        
        if not all_cifs:
            if folder_name != ".": print(f"Skipping {folder_name}: No .cif file found.")
            continue

        # 嘗試匹配您原本邏輯中的特定後綴檔案
        specific_suffix_files = [f for f in all_cifs if "_cryo2struct_full_conf_score" in f]
        
        if specific_suffix_files:
            # 優先使用符合原本命名規則的檔案
            target_cif = specific_suffix_files[0]
        else:
            # [重要改動] 如果找不到特定後綴的，就使用第一個找到的 .cif 檔
            # 這是為了避免「沒有輸出」的情況發生
            print(f"Notice: Specific suffix file not found in {folder_name}, falling back to: {all_cifs[0]}")
            target_cif = all_cifs[0]

        # 執行標記
        if target_map and target_cif:
            # 產生輸出檔名
            base_name = target_map.replace(".mrc", "")
            output_filename = f"final_atom_{base_name}.mrc"
            
            label_mask(path, target_map, target_cif, output_filename)
            
    print("All tasks complete.")