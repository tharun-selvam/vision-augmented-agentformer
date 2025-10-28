#!/usr/bin/env python3
"""
Script to create a filtered nuScenes version containing only available scenes.
Uses the nuScenes API to properly handle all relationships.
"""

import os
import sys
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import json
from collections import defaultdict
from tqdm import tqdm


def get_available_scenes(data_root):
    """
    Scan available data files and determine which scenes are complete.
    """
    print("Scanning available data files...")

    # Get all unique scene prefixes from the available files
    samples_dir = os.path.join(data_root, 'samples', 'CAM_FRONT')

    if not os.path.exists(samples_dir):
        print(f"ERROR: {samples_dir} does not exist!")
        return set()

    scene_logs = set()
    for filename in os.listdir(samples_dir):
        # Extract scene log identifier (e.g., "n008-2018-08-01-15-16-36-0400")
        scene_log = '-'.join(filename.split('__')[0].split('-')[:7])
        scene_logs.add(scene_log)

    print(f"Found {len(scene_logs)} unique scene logs:")
    for log in sorted(scene_logs):
        print(f"  - {log}")

    return scene_logs


def filter_nuscenes_metadata(version, dataroot, output_version):
    """
    Create filtered metadata using nuScenes API.
    """
    print(f"\nLoading nuScenes {version}...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # Get available scene logs
    available_logs = get_available_scenes(dataroot)

    # Find which scenes we have complete data for
    print("\nIdentifying complete scenes...")
    valid_scenes = []
    valid_scene_tokens = set()

    for scene in tqdm(nusc.scene):
        # Get the log for this scene
        log = nusc.get('log', scene['log_token'])
        log_name = log['logfile']

        # Extract log identifier
        # Log format: "n008-2018-08-01-15-16-36-0400"
        # Some have additional parts, so we match the prefix
        scene_log_match = any(log_name.startswith(avail_log) for avail_log in available_logs)

        if scene_log_match:
            # Verify by checking if first sample's data files exist
            first_sample_token = scene['first_sample_token']
            first_sample = nusc.get('sample', first_sample_token)

            # Check if LIDAR file exists
            lidar_token = first_sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join(dataroot, lidar_data['filename'])

            if os.path.exists(lidar_path):
                valid_scenes.append(scene['name'])
                valid_scene_tokens.add(scene['token'])
                print(f"  ✓ {scene['name']} (log: {log_name})")
            else:
                print(f"  ✗ {scene['name']} - missing data")

    print(f"\nFound {len(valid_scenes)} complete scenes out of {len(nusc.scene)} total")

    if len(valid_scenes) == 0:
        print("ERROR: No valid scenes found! Check your data directory.")
        return

    # Now create filtered metadata
    print("\nCreating filtered metadata...")

    # Collect all tokens we need to keep
    keep_samples = set()
    keep_sample_data = set()
    keep_annotations = set()
    keep_instances = set()
    keep_ego_poses = set()
    keep_calibrated_sensors = set()
    keep_sensors = set()
    keep_logs = set()

    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        if scene['token'] not in valid_scene_tokens:
            continue

        keep_logs.add(scene['log_token'])

        # Walk through all samples in this scene
        current_sample_token = scene['first_sample_token']

        while current_sample_token != '':
            sample = nusc.get('sample', current_sample_token)
            keep_samples.add(sample['token'])

            # Add all sample_data for this sample
            for sensor_channel, sample_data_token in sample['data'].items():
                sd = nusc.get('sample_data', sample_data_token)
                keep_sample_data.add(sd['token'])
                keep_ego_poses.add(sd['ego_pose_token'])
                keep_calibrated_sensors.add(sd['calibrated_sensor_token'])

                calib_sensor = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                keep_sensors.add(calib_sensor['sensor_token'])

                # Also keep sweep data (prev frames)
                sweep_token = sd['prev']
                while sweep_token != '':
                    sweep = nusc.get('sample_data', sweep_token)
                    keep_sample_data.add(sweep['token'])
                    keep_ego_poses.add(sweep['ego_pose_token'])
                    keep_calibrated_sensors.add(sweep['calibrated_sensor_token'])

                    sweep_calib = nusc.get('calibrated_sensor', sweep['calibrated_sensor_token'])
                    keep_sensors.add(sweep_calib['sensor_token'])

                    sweep_token = sweep['prev']

            # Add annotations
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                keep_annotations.add(ann['token'])
                keep_instances.add(ann['instance_token'])

            current_sample_token = sample['next']

    print(f"\nFiltered counts:")
    print(f"  Scenes:              {len(valid_scene_tokens)}")
    print(f"  Samples:             {len(keep_samples)}")
    print(f"  Sample Data:         {len(keep_sample_data)}")
    print(f"  Annotations:         {len(keep_annotations)}")
    print(f"  Instances:           {len(keep_instances)}")
    print(f"  Ego Poses:           {len(keep_ego_poses)}")
    print(f"  Calibrated Sensors:  {len(keep_calibrated_sensors)}")
    print(f"  Sensors:             {len(keep_sensors)}")
    print(f"  Logs:                {len(keep_logs)}")

    # Create output directory
    output_dir = os.path.join(dataroot, output_version)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nWriting filtered metadata to {output_dir}...")

    # Filter and save each table
    tables = {
        'scene': (nusc.scene, valid_scene_tokens),
        'sample': (nusc.sample, keep_samples),
        'sample_data': (nusc.sample_data, keep_sample_data),
        'sample_annotation': (nusc.sample_annotation, keep_annotations),
        'instance': (nusc.instance, keep_instances),
        'ego_pose': (nusc.ego_pose, keep_ego_poses),
        'calibrated_sensor': (nusc.calibrated_sensor, keep_calibrated_sensors),
        'sensor': (nusc.sensor, keep_sensors),
        'log': (nusc.log, keep_logs),
        'category': (nusc.category, None),
        'attribute': (nusc.attribute, None),
        'visibility': (nusc.visibility, None),
    }

    for table_name, (table_data, keep_tokens) in tables.items():
        if keep_tokens is None:
            # Keep all entries
            filtered_data = table_data
        else:
            # Filter by tokens
            filtered_data = [entry for entry in table_data if entry['token'] in keep_tokens]

        output_path = os.path.join(output_dir, f'{table_name}.json')
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f)

        print(f"  {table_name}.json: {len(filtered_data)} entries")

    # Copy map.json directly from source (contains non-serializable MapMask objects when loaded via API)
    import shutil
    src_map = os.path.join(dataroot, version, 'map.json')
    dst_map = os.path.join(output_dir, 'map.json')
    shutil.copy(src_map, dst_map)
    print(f"  map.json: copied from source")

    print(f"\n✓ Filtering complete!")
    print(f"✓ Filtered metadata saved to: {output_dir}")
    print(f"\nValid scenes list:")
    for scene_name in sorted(valid_scenes):
        print(f"  - {scene_name}")


def main():
    parser = argparse.ArgumentParser(description='Filter nuScenes metadata to match available data')
    parser.add_argument('--dataroot', type=str,
                        default='/workspace/vision-augmented-agentformer/AgentFormer/nuscenes',
                        help='Path to nuScenes data directory')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version to filter')
    parser.add_argument('--output_version', type=str, default='v1.0-trainval-subset',
                        help='Output version name')

    args = parser.parse_args()

    print("="*60)
    print("nuScenes Metadata Filter (API-based)")
    print("="*60)
    print(f"Data root:      {args.dataroot}")
    print(f"Input version:  {args.version}")
    print(f"Output version: {args.output_version}")
    print("="*60)

    filter_nuscenes_metadata(args.version, args.dataroot, args.output_version)


if __name__ == '__main__':
    main()
