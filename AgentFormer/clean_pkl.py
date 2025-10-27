
import pickle
import os
from tqdm import tqdm

def clean_pkl_file(pkl_path, data_root):
    print(f"Loading pkl file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Found {len(data)} total samples.")
    good_samples = []
    for sample in tqdm(data):
        all_files_exist = True
        # Check lidar data
        if 'lidar_infos' in sample and 'LIDAR_TOP' in sample['lidar_infos']:
            lidar_path = os.path.join(data_root, sample['lidar_infos']['LIDAR_TOP']['filename'])
            if not os.path.exists(lidar_path):
                all_files_exist = False
                # print(f"Missing Lidar: {lidar_path}")

        # Check camera data
        if all_files_exist and 'cam_infos' in sample:
            for cam, cam_info in sample['cam_infos'].items():
                img_path = os.path.join(data_root, cam_info['filename'])
                if not os.path.exists(img_path):
                    all_files_exist = False
                    # print(f"Missing Image: {img_path}")
                    break

        if all_files_exist:
            good_samples.append(sample)

    print(f"Found {len(good_samples)} valid samples.")
    print(f"Saving cleaned data back to {pkl_path}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(good_samples, f)

if __name__ == '__main__':
    data_root = '/home/tharun/Documents/BTP/AgentFormer/nuscenes'
    train_pkl = '/home/tharun/Documents/BTP/AgentFormer/datasets/nuscenes_pred/nuscenes_infos_train.pkl'
    val_pkl = '/home/tharun/Documents/BTP/AgentFormer/datasets/nuscenes_pred/nuscenes_infos_val.pkl'
    clean_pkl_file(train_pkl, data_root)
    clean_pkl_file(val_pkl, data_root)
