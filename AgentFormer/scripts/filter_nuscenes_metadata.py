#!/usr/bin/env python3
"""
Script to filter nuScenes metadata to match only the available data files.
This is useful when you have a partial dataset (e.g., only part 1 of 10).
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def get_available_filenames(data_root):
    """Scan the data directory to find all available sensor files."""
    print("Scanning available data files...")

    available_files = set()

    # Scan samples directory
    samples_dir = os.path.join(data_root, 'samples')
    if os.path.exists(samples_dir):
        for sensor_type in os.listdir(samples_dir):
            sensor_path = os.path.join(samples_dir, sensor_type)
            if os.path.isdir(sensor_path):
                for filename in os.listdir(sensor_path):
                    rel_path = f'samples/{sensor_type}/{filename}'
                    available_files.add(rel_path)

    # Scan sweeps directory
    sweeps_dir = os.path.join(data_root, 'sweeps')
    if os.path.exists(sweeps_dir):
        for sensor_type in os.listdir(sweeps_dir):
            sensor_path = os.path.join(sweeps_dir, sensor_type)
            if os.path.isdir(sensor_path):
                for filename in os.listdir(sensor_path):
                    rel_path = f'sweeps/{sensor_type}/{filename}'
                    available_files.add(rel_path)

    print(f"Found {len(available_files)} available data files")
    return available_files


def filter_metadata(metadata_dir, output_dir, available_files, data_root):
    """Filter all metadata JSON files to only include available data."""

    print(f"\nFiltering metadata from {metadata_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load all metadata files
    metadata = {}
    for json_file in ['sample.json', 'sample_data.json', 'sample_annotation.json',
                      'instance.json', 'scene.json', 'log.json', 'ego_pose.json',
                      'sensor.json', 'calibrated_sensor.json', 'category.json',
                      'attribute.json', 'visibility.json', 'map.json']:
        filepath = os.path.join(metadata_dir, json_file)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                metadata[json_file] = json.load(f)
            print(f"Loaded {json_file}: {len(metadata[json_file])} entries")

    # Step 1: Filter sample_data based on available files
    print("\nFiltering sample_data...")
    valid_sample_data_tokens = set()
    filtered_sample_data = []

    for sd in metadata['sample_data.json']:
        if sd['filename'] in available_files:
            valid_sample_data_tokens.add(sd['token'])
            filtered_sample_data.append(sd)

    print(f"  Kept {len(filtered_sample_data)} / {len(metadata['sample_data.json'])} sample_data entries")

    # Step 2: Filter samples that have all required sensor data
    print("\nFiltering samples...")
    valid_sample_tokens = set()
    filtered_samples = []

    # Build sample_data lookup by sample token
    sample_to_data = defaultdict(list)
    for sd in filtered_sample_data:
        if sd['sample_token'] != '':  # Not a sweep
            sample_to_data[sd['sample_token']].append(sd['token'])

    for sample in metadata['sample.json']:
        # Check if the key frames exist
        data_tokens = []
        current = sample['data']

        # Check all sensor data for this sample
        all_exist = True
        for sensor_name, data_token in current.items():
            if data_token not in valid_sample_data_tokens:
                all_exist = False
                break

        if all_exist:
            valid_sample_tokens.add(sample['token'])
            filtered_samples.append(sample)

    print(f"  Kept {len(filtered_samples)} / {len(metadata['sample.json'])} sample entries")

    # Step 3: Filter sample_annotations
    print("\nFiltering sample_annotations...")
    valid_annotation_tokens = set()
    valid_instance_tokens = set()
    filtered_annotations = []

    for ann in metadata['sample_annotation.json']:
        if ann['sample_token'] in valid_sample_tokens:
            valid_annotation_tokens.add(ann['token'])
            valid_instance_tokens.add(ann['instance_token'])
            filtered_annotations.append(ann)

    print(f"  Kept {len(filtered_annotations)} / {len(metadata['sample_annotation.json'])} annotation entries")

    # Step 4: Filter instances
    print("\nFiltering instances...")
    filtered_instances = []
    for inst in metadata['instance.json']:
        if inst['token'] in valid_instance_tokens:
            filtered_instances.append(inst)

    print(f"  Kept {len(filtered_instances)} / {len(metadata['instance.json'])} instance entries")

    # Step 5: Filter scenes
    print("\nFiltering scenes...")
    valid_scene_tokens = set()
    valid_log_tokens = set()
    filtered_scenes = []

    for scene in metadata['scene.json']:
        # Check if first_sample_token is valid
        if scene['first_sample_token'] in valid_sample_tokens:
            valid_scene_tokens.add(scene['token'])
            valid_log_tokens.add(scene['log_token'])
            filtered_scenes.append(scene)

    print(f"  Kept {len(filtered_scenes)} / {len(metadata['scene.json'])} scene entries")

    # Step 6: Filter logs
    print("\nFiltering logs...")
    filtered_logs = []
    for log in metadata['log.json']:
        if log['token'] in valid_log_tokens:
            filtered_logs.append(log)

    print(f"  Kept {len(filtered_logs)} / {len(metadata['log.json'])} log entries")

    # Step 7: Filter ego_pose and calibrated_sensor based on sample_data
    print("\nFiltering ego_pose...")
    valid_ego_pose_tokens = set()
    valid_calib_tokens = set()

    for sd in filtered_sample_data:
        valid_ego_pose_tokens.add(sd['ego_pose_token'])
        valid_calib_tokens.add(sd['calibrated_sensor_token'])

    filtered_ego_pose = []
    for ep in metadata['ego_pose.json']:
        if ep['token'] in valid_ego_pose_tokens:
            filtered_ego_pose.append(ep)

    print(f"  Kept {len(filtered_ego_pose)} / {len(metadata['ego_pose.json'])} ego_pose entries")

    print("\nFiltering calibrated_sensor...")
    valid_sensor_tokens = set()
    filtered_calib = []
    for cs in metadata['calibrated_sensor.json']:
        if cs['token'] in valid_calib_tokens:
            valid_sensor_tokens.add(cs['sensor_token'])
            filtered_calib.append(cs)

    print(f"  Kept {len(filtered_calib)} / {len(metadata['calibrated_sensor.json'])} calibrated_sensor entries")

    # Step 8: Filter sensor
    print("\nFiltering sensor...")
    filtered_sensor = []
    for sensor in metadata['sensor.json']:
        if sensor['token'] in valid_sensor_tokens:
            filtered_sensor.append(sensor)

    print(f"  Kept {len(filtered_sensor)} / {len(metadata['sensor.json'])} sensor entries")

    # Step 9: Keep all category, attribute, visibility, map (these are small and universal)
    print("\nCopying universal metadata (category, attribute, visibility, map)...")

    # Save all filtered metadata
    filtered_metadata = {
        'sample.json': filtered_samples,
        'sample_data.json': filtered_sample_data,
        'sample_annotation.json': filtered_annotations,
        'instance.json': filtered_instances,
        'scene.json': filtered_scenes,
        'log.json': filtered_logs,
        'ego_pose.json': filtered_ego_pose,
        'sensor.json': filtered_sensor,
        'calibrated_sensor.json': filtered_calib,
        'category.json': metadata['category.json'],
        'attribute.json': metadata['attribute.json'],
        'visibility.json': metadata['visibility.json'],
        'map.json': metadata['map.json']
    }

    print("\nWriting filtered metadata files...")
    for filename, data in filtered_metadata.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f)
        print(f"  Wrote {output_path}: {len(data)} entries")

    # Print summary
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Scenes:       {len(filtered_scenes)} / {len(metadata['scene.json'])} ({100*len(filtered_scenes)/len(metadata['scene.json']):.1f}%)")
    print(f"Samples:      {len(filtered_samples)} / {len(metadata['sample.json'])} ({100*len(filtered_samples)/len(metadata['sample.json']):.1f}%)")
    print(f"Sample Data:  {len(filtered_sample_data)} / {len(metadata['sample_data.json'])} ({100*len(filtered_sample_data)/len(metadata['sample_data.json']):.1f}%)")
    print(f"Annotations:  {len(filtered_annotations)} / {len(metadata['sample_annotation.json'])} ({100*len(filtered_annotations)/len(metadata['sample_annotation.json']):.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Filter nuScenes metadata to match available data')
    parser.add_argument('--data_root', type=str,
                        default='/workspace/vision-augmented-agentformer/AgentFormer/nuscenes',
                        help='Path to nuScenes data directory')
    parser.add_argument('--metadata_version', type=str, default='v1.0-trainval',
                        help='Metadata version to filter (e.g., v1.0-trainval)')
    parser.add_argument('--output_version', type=str, default='v1.0-trainval-subset',
                        help='Output metadata version name')

    args = parser.parse_args()

    data_root = args.data_root
    metadata_dir = os.path.join(data_root, args.metadata_version)
    output_dir = os.path.join(data_root, args.output_version)

    print("="*60)
    print("nuScenes Metadata Filter")
    print("="*60)
    print(f"Data root:      {data_root}")
    print(f"Input metadata: {metadata_dir}")
    print(f"Output:         {output_dir}")
    print("="*60)

    # Step 1: Get available files
    available_files = get_available_filenames(data_root)

    # Step 2: Filter metadata
    filter_metadata(metadata_dir, output_dir, available_files, data_root)

    print("\n✓ Filtering complete!")
    print(f"✓ Filtered metadata saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Update your config files to use version '{args.output_version}'")
    print(f"2. Regenerate the .pkl info files using gen_info.py")


if __name__ == '__main__':
    main()
