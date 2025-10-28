#!/usr/bin/env python3
"""
Generate .pkl info files for the filtered nuScenes subset.
Based on BEVDepth's gen_info.py but adapted for the v1.0-trainval-subset.
"""

import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import argparse
import os


def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=10):
    """
    Generate info dict for each sample in the specified scenes.
    """
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            sweep_cam_info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        if sweep_lidar_info['sample_token'] != lidar_data[
                                'sample_token']:
                            break
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            ann_infos = list()
            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def get_available_scenes(nusc):
    """
    Get list of all scene names in the loaded dataset.
    """
    return [scene['name'] for scene in nusc.scene]


def main():
    parser = argparse.ArgumentParser(description='Generate nuScenes info files for subset')
    parser.add_argument('--dataroot', type=str,
                        default='/workspace/vision-augmented-agentformer/AgentFormer/nuscenes',
                        help='Path to nuScenes data directory')
    parser.add_argument('--version', type=str, default='v1.0-trainval-subset',
                        help='nuScenes version to use')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for .pkl files (default: dataroot)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.dataroot

    print("="*60)
    print("nuScenes Info Generator (Subset)")
    print("="*60)
    print(f"Data root:   {args.dataroot}")
    print(f"Version:     {args.version}")
    print(f"Output dir:  {args.output_dir}")
    print("="*60)

    # Load the subset
    print(f"\nLoading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Get all available scenes
    all_scenes = get_available_scenes(nusc)
    print(f"\nFound {len(all_scenes)} scenes in {args.version}")

    # For subset, we typically want to use all scenes for both train and val
    # since we have limited data. But we can split them if needed.
    # Let's check which scenes overlap with official train/val splits

    train_overlap = [s for s in all_scenes if s in splits.train]
    val_overlap = [s for s in all_scenes if s in splits.val]

    print(f"\nScenes overlapping with official train split: {len(train_overlap)}")
    print(f"Scenes overlapping with official val split: {len(val_overlap)}")

    if len(train_overlap) > 0:
        print("\nGenerating train info...")
        train_infos = generate_info(nusc, train_overlap)
        output_path = os.path.join(args.output_dir, f'nuscenes_infos_train_subset.pkl')
        mmengine.dump(train_infos, output_path)
        print(f"✓ Saved {len(train_infos)} samples to {output_path}")

    if len(val_overlap) > 0:
        print("\nGenerating val info...")
        val_infos = generate_info(nusc, val_overlap)
        output_path = os.path.join(args.output_dir, f'nuscenes_infos_val_subset.pkl')
        mmengine.dump(val_infos, output_path)
        print(f"✓ Saved {len(val_infos)} samples to {output_path}")

    # Also generate a combined file using all scenes
    print("\nGenerating combined info (all scenes)...")
    all_infos = generate_info(nusc, all_scenes)
    output_path = os.path.join(args.output_dir, f'nuscenes_infos_all_subset.pkl')
    mmengine.dump(all_infos, output_path)
    print(f"✓ Saved {len(all_infos)} samples to {output_path}")

    print("\n" + "="*60)
    print("Info generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
