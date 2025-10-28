#!/usr/bin/env python3
"""
Parallel version of process_nuscenes.py for faster preprocessing.
Uses multiprocessing to process scenes in parallel across CPU cores.

Expected speedup: 5-10x faster (85 scenes in ~10-20 minutes vs 1-2 hours)
"""

import json
import os
from pyquaternion import Quaternion
from itertools import chain
from typing import List, Tuple
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
import cv2
import argparse
import gc
from multiprocessing import Pool, cpu_count
from functools import partial

NUM_IN_TRAIN_VAL = 200
past_frames = 4
future_frames = 12


def process_single_scene(scene_info: Tuple, DATAROOT: str, DATAOUT: str, map_version: str) -> Tuple[str, bool, str]:
    """
    Process a single scene independently.

    Args:
        scene_info: Tuple of (scene_dict, split_name, scene_name)
        DATAROOT: Path to nuScenes dataset
        DATAOUT: Output directory
        map_version: Map version string

    Returns:
        Tuple of (scene_name, success, error_message)
    """
    scene, split, scene_name = scene_info

    try:
        # Each worker needs its own NuScenes instance
        nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT, verbose=False)
        helper = PredictHelper(nuscenes)

        # Load prediction scenes map
        prediction_scenes_map = json.load(open(os.path.join(DATAROOT, "maps", "prediction", "prediction_scenes.json"), "r"))

        scene_token = scene['token']
        scene_data_orig = prediction_scenes_map.get(scene_name, [])
        if len(scene_data_orig) == 0:
            return (scene_name, False, "No prediction data")

        scene_data_orig_set = set(scene_data_orig)
        scene_data = set(scene_data_orig)

        # Expand scene data with past and future frames
        for data in scene_data_orig:
            cur_sample = helper.get_sample_annotation(*data.split('_'))
            sample = cur_sample
            for i in range(past_frames - 1):
                if sample['prev'] == '':
                    break
                sample = nuscenes.get('sample_annotation', sample['prev'])
                cur_data = sample['instance_token'] + '_' + sample['sample_token']
                scene_data.add(cur_data)
            sample = cur_sample
            for i in range(future_frames):
                if sample['next'] == '':
                    break
                sample = nuscenes.get('sample_annotation', sample['next'])
                cur_data = sample['instance_token'] + '_' + sample['sample_token']
                scene_data.add(cur_data)

        all_tokens = np.array([x.split("_") for x in scene_data])
        all_samples = set(np.unique(all_tokens[:, 1]).tolist())
        all_instances = np.unique(all_tokens[:, 0]).tolist()
        first_sample_token = scene['first_sample_token']
        first_sample = nuscenes.get('sample', first_sample_token)
        while first_sample['token'] not in all_samples and first_sample['next'] != '':
            first_sample = nuscenes.get('sample', first_sample['next'])
        if first_sample['token'] not in all_samples:
            return (scene_name, False, "No valid samples found")

        frame_id = 0
        sample = first_sample
        cvt_data = []
        while True:
            if sample['token'] in all_samples:
                instances_in_frame = []
                for ann_token in sample['anns']:
                    annotation = nuscenes.get('sample_annotation', ann_token)
                    category = annotation['category_name']
                    instance = annotation['instance_token']
                    cur_data = instance + '_' + annotation['sample_token']
                    if cur_data not in scene_data:
                        continue
                    instances_in_frame.append(instance)
                    # get data
                    data = np.ones(18) * -1.0
                    data[0] = frame_id
                    data[1] = all_instances.index(instance)
                    data[10] = annotation['size'][0]
                    data[11] = annotation['size'][2]
                    data[12] = annotation['size'][1]
                    data[13] = annotation['translation'][0]
                    data[14] = annotation['translation'][2]
                    data[15] = annotation['translation'][1]
                    data[16] = Quaternion(annotation['rotation']).yaw_pitch_roll[0]
                    data[17] = 1 if cur_data in scene_data_orig_set else 0
                    data = data.astype(str)
                    if 'car' in category:
                        data[2] = 'Car'
                    elif 'bus' in category:
                        data[2] = 'Bus'
                    elif 'truck' in category:
                        data[2] = 'Truck'
                    elif 'emergency' in category:
                        data[2] = 'Emergency'
                    elif 'construction' in category:
                        data[2] = 'Construction'
                    else:
                        # Skip vehicle categories not handled by the model
                        continue
                    cvt_data.append(data)

            frame_id += 1
            if sample['next'] != '':
                sample = nuscenes.get('sample', sample['next'])
            else:
                break

        if not cvt_data:
            return (scene_name, False, "No valid trajectory data")

        cvt_data = np.stack(cvt_data)

        # Generate Maps
        map_name = nuscenes.get('log', scene['log_token'])['location']
        nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=map_name)
        scale = 3.0
        margin = 75
        xy = cvt_data[:, [13, 15]].astype(np.float32)
        x_min = np.round(xy[:, 0].min() - margin)
        x_max = np.round(xy[:, 0].max() + margin)
        y_min = np.round(xy[:, 1].min() - margin)
        y_max = np.round(xy[:, 1].max() + margin)
        x_size = x_max - x_min
        y_size = y_max - y_min
        patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
        patch_angle = 0
        canvas_size = (np.round(scale * y_size).astype(int), np.round(scale * x_size).astype(int))
        homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line', 'ped_crossing', 'walkway']
        colors = {
            'rest': [255, 240, 243],
            'lane': [206, 229, 223],
            'road_segment': [206, 229, 223],
            'drivable_area': [206, 229, 223],
            'ped_crossing': [226, 228, 234],
            'walkway': [169, 209, 232],
            'road_divider': [255, 251, 242],
            'lane_divider': [100, 100, 100],
            'stop_line': [0, 255, 255],
        }

        map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(np.uint8)
        map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
        map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)

        # map for visualization
        map_mask_plot = np.ones_like(map_mask[:3])
        map_mask_plot[:] = np.array(colors['rest'])[:, None, None]
        for layer in ['lane', 'road_segment', 'drivable_area', 'road_divider', 'ped_crossing', 'walkway']:
            xind, yind = np.where(map_mask[layer_names.index(layer)])
            map_mask_plot[:, xind, yind] = np.array(colors[layer])[:, None]

        # Save files
        meta = np.array([x_min, y_min, scale])
        np.savetxt(f'{DATAOUT}/map_{map_version}/meta_{scene_name}.txt', meta, fmt='%.2f')
        cv2.imwrite(f'{DATAOUT}/map_{map_version}/{scene_name}.png', np.transpose(map_mask_vehicle, (1, 2, 0)))
        cv2.imwrite(f'{DATAOUT}/map_{map_version}/vis_{scene_name}.png', cv2.cvtColor(np.transpose(map_mask_plot, (1, 2, 0)), cv2.COLOR_RGB2BGR))

        pred_num = int(cvt_data[:, -1].astype(np.float32).sum())
        assert pred_num == len(scene_data_orig), f"Prediction count mismatch: {pred_num} vs {len(scene_data_orig)}"

        np.savetxt(f'{DATAOUT}/label/{split}/{scene_name}.txt', cvt_data, fmt='%s')

        # Clean up
        del nuscenes, helper, nusc_map
        gc.collect()

        return (scene_name, True, f"map_shape {map_mask_plot.shape}")

    except Exception as e:
        return (scene_name, False, f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process nuScenes dataset in parallel')
    parser.add_argument('--data_root', required=True, help="path to the original nuScenes dataset")
    parser.add_argument('--data_out', default='datasets/nuscenes_pred/', help="path where you save the processed data")
    parser.add_argument('--num_workers', type=int, default=None, help="number of parallel workers (default: CPU count - 2)")
    args = parser.parse_args()

    DATAROOT = args.data_root
    DATAOUT = args.data_out
    map_version = '0.1'

    print("="*80)
    print("Parallel nuScenes Preprocessing")
    print("="*80)
    print(f"Data root:    {DATAROOT}")
    print(f"Output dir:   {DATAOUT}")
    print(f"Map version:  {map_version}")

    # Determine number of workers
    if args.num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free
    else:
        num_workers = args.num_workers
    print(f"CPU cores:    {cpu_count()} total, using {num_workers} workers")
    print("="*80)

    # Load nuScenes dataset (main process only, for scene enumeration)
    print("\nLoading nuScenes dataset...")
    nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT, verbose=True)

    # Create output directories
    print("Creating output directories...")
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{DATAOUT}/label/{split}', exist_ok=True)
    os.makedirs(f'{DATAOUT}/map_{map_version}', exist_ok=True)

    # Get the official splits to map scene names to the correct folder
    official_splits = create_splits_scenes()

    # Create a reverse map from scene name to split name
    scene_to_split = {}
    for split_name, scene_list in official_splits.items():
        for scene_name in scene_list:
            if split_name in ['train', 'val']:
                scene_to_split[scene_name] = split_name

    # Prepare scene list for processing
    print("\nPreparing scene list...")
    scene_infos = []
    for scene in nuscenes.scene:
        scene_name = scene['name']

        # Determine which split this scene belongs to
        if scene_name in scene_to_split:
            # Adjust for the special 'test' case which uses 'val' scenes
            if scene_to_split[scene_name] == 'val' and scene_name in official_splits['val']:
                split = 'test'
            else:
                split = scene_to_split[scene_name]
        else:
            print(f"Skipping scene {scene_name} (not in train or val splits)")
            continue

        scene_infos.append((scene, split, scene_name))

    print(f"Found {len(scene_infos)} scenes to process")
    print(f"Estimated time: ~{len(scene_infos) // num_workers} minutes (with {num_workers} workers)")
    print("\nProcessing scenes in parallel...\n")

    # Process scenes in parallel
    process_func = partial(process_single_scene,
                          DATAROOT=DATAROOT,
                          DATAOUT=DATAOUT,
                          map_version=map_version)

    with Pool(num_workers) as pool:
        results = pool.map(process_func, scene_infos)

    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count

    print(f"Total scenes: {len(results)}")
    print(f"Successful:   {success_count}")
    print(f"Failed:       {failed_count}")

    if failed_count > 0:
        print("\nFailed scenes:")
        for scene_name, success, msg in results:
            if not success:
                print(f"  - {scene_name}: {msg}")

    print("\nOutput files:")
    print(f"  - Labels:     {DATAOUT}/label/{{train,val,test}}/scene-*.txt")
    print(f"  - Maps:       {DATAOUT}/map_{map_version}/{{scene-*.png,meta_scene-*.txt}}")
    print("="*80)
