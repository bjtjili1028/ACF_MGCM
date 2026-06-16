import mrcfile
import numpy as np

def write_centroid_file(centroids, out_path):
    """將質心寫入文本文件"""
    if out_path:
        with open(out_path, 'w') as f:
            for x, y, z, prob in centroids:
                f.write(f'{x:.3f} {y:.3f} {z:.3f} {prob:.4f}\n')
        print(f"寫入 {len(centroids)} 個質心到 {out_path}")

def write_mrc(ca_centroids, n_centroids, c_centroids, ref_map, out_path):
    """將質心點寫入新的 MRC 檔案"""
    print(f"\n寫入MRC文件到 {out_path}...")
    
    try:
        with mrcfile.open(ref_map) as mrc_ref:
            ref_header = mrc_ref.header
            ref_voxel_size = mrc_ref.voxel_size
            new_data = np.zeros(mrc_ref.data.shape, dtype=np.float32)
    except Exception as e:
        print(f"錯誤: 無法讀取參考 MRC 檔案 '{ref_map}'. {e}")
        return

    def place_atoms(centroids, value):
        if not centroids: return
        count = 0
        origin = np.array([ref_header.origin.x, ref_header.origin.y, ref_header.origin.z])
        voxel_size = ref_voxel_size.x
        
        for x, y, z, prob in centroids:
            ix = int(round((x - origin[0]) / voxel_size))
            iy = int(round((y - origin[1]) / voxel_size))
            iz = int(round((z - origin[2]) / voxel_size))
            
            if 0 <= iz < new_data.shape[0] and 0 <= iy < new_data.shape[1] and 0 <= ix < new_data.shape[2]:
                new_data[iz, iy, ix] = value
                count += 1
        print(f"  - 成功標記 {count} 個 {['', 'CA', 'N', 'C'][value]} 原子")

    place_atoms(ca_centroids, 1)
    place_atoms(n_centroids, 2)
    place_atoms(c_centroids, 3)
    
    try:
        with mrcfile.new(out_path, overwrite=True) as mrc_new:
            mrc_new.set_data(new_data)
            mrc_new.voxel_size = ref_voxel_size
            mrc_new.header.origin = ref_header.origin
        print(f"成功寫入檔案: {out_path}")
    except Exception as e:
        print(f"錯誤: 無法寫入新的 MRC 檔案 '{out_path}'. {e}")