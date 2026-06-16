"""
@author: nabin

This script labels CA to 1, N to 2 and C to 3
此腳本將 PDB 檔案中的 CA 原子標記為 1，N 原子標記為 2，C 原子標記為 3，並儲存為 MRC 格式的標記圖。

"""
import sys

import mrcfile
import math
import numpy as np
from Bio import PDB
import os

error_list = set()

# 根據座標和原始地圖的原點、體素大小計算索引值

def get_index(cord, origin, voxel):
    # return math.ceil(math.floor(cord - origin) / voxel)  # 向下取整 (round down) 原始使用
    # return math.ceil((cord - origin) / voxel)   # 向上取整 (round up)
    # return math.floor(cord - origin / voxel)  # 向下取整 (round down)
    return round((cord - origin) / voxel)  # 四捨五入 (round off)

# 僅將CA原子標記為1並儲存為MRC檔案
def ca_mask(path, org_map, pdb_map, outfilename):
    count = 0
    org_map = os.path.join(path, org_map)
    org_map = mrcfile.open(org_map, mode='r')
    data = np.zeros(org_map.data.shape, dtype=np.int16)
    data = data.astype('float32')

    # 獲取原始資料的座標原點與體素大小
    x_origin = org_map.header.origin['x']
    y_origin = org_map.header.origin['y']
    z_origin = org_map.header.origin['z']
    x_voxel = org_map.voxel_size['x']
    y_voxel = org_map.voxel_size['y']
    z_voxel = org_map.voxel_size['z']
    
    # 使用 BioPython 的PDB解析器來讀取 pdb 結構資料
    parser = PDB.PDBParser()
    pdb_map = os.path.join(path, pdb_map)
    struct = parser.get_structure("CA", pdb_map)
    
    # 逐一檢查所有原子座標
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.get_coord()
                    iz = int(get_index(z, z_origin, z_voxel))
                    jy = int(get_index(y, y_origin, y_voxel))
                    kx = int(get_index(x, x_origin, x_voxel))
                    
                    # 如果是CA原子，則標記為1
                    if atom.get_name() == "CA":
                        try:
                            data[iz, jy, kx] = 1
                            count += 1
                        except IndexError as error:
                            error_list.add(pdb_map)
                            
    print("Saving the file - - - - - ", org_map)
    outfilename = path + "/" + outfilename
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        mrc.header.origin = org_map.header.origin
        mrc.close()
    print(outfilename, "Done")
    print(f"Number of Carbon Alpha in label map is => {count}")

# 將CA, N, C原子分別標記為1, 2, 3並儲存為MRC檔案
def label_mask(path, org_map, pdb_map, outfilename):
    count = 0
    org_map = os.path.join(path, org_map)
    org_map = mrcfile.open(org_map, mode='r')
    data = np.zeros(org_map.data.shape, dtype=np.int16)
    data = data.astype('float32')
    
    # 原始地圖資訊
    x_origin = org_map.header.origin['x']
    y_origin = org_map.header.origin['y']
    z_origin = org_map.header.origin['z']
    x_voxel = org_map.voxel_size['x']
    y_voxel = org_map.voxel_size['y']
    z_voxel = org_map.voxel_size['z']
    
    # 讀取PDB檔案
    parser = PDB.PDBParser()
    pdb_map = os.path.join(path, pdb_map)
    struct = parser.get_structure("CA", pdb_map)
    
    # 逐一檢查所有原子並分別標記CA=1, N=2, C=3
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.get_coord()
                    iz = int(get_index(z, z_origin, z_voxel))
                    jy = int(get_index(y, y_origin, y_voxel))
                    kx = int(get_index(x, x_origin, x_voxel))
                    
                    if atom.get_name() == "CA":
                        try:
                            data[iz, jy, kx] = 1
                            count += 1
                        except IndexError as error:
                            error_list.add(pdb_map)
                    elif atom.get_name() == "N":
                        try:
                            data[iz, jy, kx] = 2
                        except IndexError as error:
                            error_list.add(pdb_map)
                    elif atom.get_name() == "C":
                        try:
                            data[iz, jy, kx] = 3
                        except IndexError as error:
                            error_list.add(pdb_map)
                            
    print("Saving the file - - - - - ", org_map)
    outfilename = path + "/" + outfilename
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        mrc.header.origin = org_map.header.origin
        mrc.close()
    print(outfilename, "Done")
    print(f"Number of Carbon Alpha in label map is => {count}")


# 主程式區塊
if __name__ == "__main__":
    input_path = sys.argv[1]  # 接收命令列參數作為輸入路徑
    undone_pdb_emd = list()  # 儲存未處理的資料夾名稱
    map_names = [fn for fn in os.listdir(input_path) if not fn.endswith(".pdb")]  # 列出所有非PDB檔案的資料夾
    
    print("########### Generating atoms label ##########")
    for _ in range(len(map_names)):
        path = os.path.join(input_path) # , map_names[_]
        emd_map = [e for e in os.listdir(path) if e.endswith(".mrc")]
        print("emd_map:",emd_map)
        path = os.path.join(input_path) # , map_names[_]
        pdb_map = [p for p in os.listdir(path) if p.endswith(".pdb")]
        pdb_map.sort()
        pdb_map = pdb_map[0].split(".")[0]
        pdb_map = pdb_map.split("_")[0]
        pdb_map = pdb_map.lower()
        pdb_map = pdb_map + "_cryo2struct_full_conf_score.pdb"  # v1
        # pdb_map = pdb_map + "_cryo2struct_full_conf_score_3.pdb"  # v2
        # pdb_map = pdb_map + "_clusters_pre_hmm.pdb"  # cluster
        # pdb_map = pdb_map + ".pdb"  # cryo_atom
        print("emd_map:",pdb_map)
        
        # 確認資料夾內同時存在pdb與emd檔案才執行
        if len(pdb_map) != 0 and len(emd_map) != 0:
            print("Working on => ", pdb_map, " of => ", map_names[_])
            em = "emd_normalized_map.mrc"
            name = em.split(".")
            # atoms labeling
            label_mask(path, em, pdb_map, "final_atom_" + name[0] + ".mrc")  # final pdb
            # label_mask(path, em, pdb_map, "cluster_atom_" + name[0] + ".mrc") # cluster pdb
            # ca only labeling
            # ca_mask(path, em, pdb_map, "round_off_atom_ca_" + name[0] + ".mrc")
        else:
            undone_pdb_emd.append(map_names[_]) # 記錄未處理的檔案夾

    print("Atoms and Ca only labeling Complete!")

