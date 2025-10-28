from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy
from data.bev_utils import *

# Optional BEVDepth import - only needed for on-the-fly BEV feature extraction
try:
    from bevdepth.datasets.nusc_det_dataset import NuscDetDataset
    BEVDEPTH_AVAILABLE = True
except ImportError:
    NuscDetDataset = None
    BEVDEPTH_AVAILABLE = False

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from utils.utils import print_log


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training'):
        self.parser = parser
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        # Check if we should use BEV features (hybrid approach)
        self.use_bev = parser.get('use_bev', False)
        self.nusc_det_dataset = None

        # Initialize BEV dataset if needed (for image features)
        # Only needed if NOT using pre-computed features
        use_precomputed = parser.get('use_precomputed_bev', False)
        if parser.dataset == 'nuscenes_pred' and self.use_bev and not use_precomputed:
            if not BEVDEPTH_AVAILABLE:
                raise ImportError(
                    "BEVDepth is required for on-the-fly BEV feature extraction. "
                    "Either:\n"
                    "1. Install BEVDepth: git clone https://github.com/Megvii-BaseDetection/BEVDepth.git bevdepth\n"
                    "2. Use pre-computed features: Set 'use_precomputed_bev: true' in your config\n"
                    "3. Disable BEV: Set 'use_bev: false' for baseline mode"
                )

            data_root = parser.data_root_nuscenes_pred
            # Use subset pkl files if they exist, otherwise fall back to full dataset
            subset_pkl_path = os.path.join(data_root, f'nuscenes_infos_{split}_subset.pkl')
            if os.path.exists(subset_pkl_path):
                info_paths = subset_pkl_path
                print_log(f'Initializing BEV dataset for image features with subset: {info_paths}', log)
            else:
                info_paths = os.path.join(data_root, f'nuscenes_infos_{split}.pkl')
                print_log(f'Initializing BEV dataset for image features with full dataset: {info_paths}', log)

            # Determine the correct nuscenes data root
            # data_root is 'datasets/nuscenes_pred', we need 'nuscenes'
            nuscenes_data_root = 'nuscenes'

            self.nusc_det_dataset = NuscDetDataset(
                ida_aug_conf=parser.ida_aug_conf,
                bda_aug_conf=parser.bda_aug_conf,
                classes=parser.object_classes,
                data_root=nuscenes_data_root,
                info_paths=info_paths,
                is_train=(split == 'train'),
                img_conf=parser.img_conf,
                num_sweeps=1,
                return_depth=True,
            )
        elif parser.dataset == 'nuscenes_pred' and self.use_bev and use_precomputed:
            print_log(f'Using pre-computed BEV features (BEVDepth not required)', log)

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
            print_log(f'Using original AgentFormer dataloader for nuScenes', log)
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy            
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]

        # Get trajectory data from AgentFormer preprocessor
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        data = seq(frame)

        # Augment with BEV data if enabled
        if data is not None and self.use_bev and self.nusc_det_dataset is not None:
            # Get corresponding BEV image data (approximate matching by sample index)
            bev_sample_idx = sample_index % len(self.nusc_det_dataset)
            bev_data = self.nusc_det_dataset[bev_sample_idx]

            # bev_data is a list: [sweep_imgs, sensor2ego_mats, intrins, ida_mats, sensor2sensor_mats, bda_mat, timestamps, img_metas, gt_boxes, gt_labels]
            img_metas = bev_data[7]
            sample_token = img_metas['token']

            # Check if pre-computed BEV features exist
            use_precomputed = self.parser.get('use_precomputed_bev', False)
            precomputed_path = None

            if use_precomputed:
                # Try to load pre-computed features
                feature_dir = os.path.join('bev_features', f'{self.split}_subset' if 'subset' in self.split else self.split)
                feature_file = os.path.join(feature_dir, f'{sample_token}.pt')

                if os.path.exists(feature_file):
                    precomputed_path = feature_file

            if precomputed_path is not None:
                # Load pre-computed BEV feature map
                import torch
                data['bev_feature_map'] = torch.load(precomputed_path)
                data['sample_token'] = sample_token
                # Don't load images when using pre-computed features (save memory)
            else:
                # Load raw images for on-the-fly BEV extraction
                data['sweep_imgs'] = bev_data[0]
                data['mats_dict'] = {
                    'sensor2ego_mats': bev_data[1],
                    'intrin_mats': bev_data[2],
                    'ida_mats': bev_data[3],
                    'sensor2sensor_mats': bev_data[4],
                    'bda_mat': bev_data[5],
                }
                data['timestamps'] = bev_data[6]
                data['img_metas'] = bev_data[7]

        self.index += 1
        return data      

    def __call__(self):
        return self.next_sample()
