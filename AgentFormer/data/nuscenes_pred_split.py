import glob
import os


def get_nuscenes_pred_split(data_root):
     # List of 85 scenes from the 10% subset (Part 1 of nuScenes)
     # These match the scenes we have complete data for
     subset_scenes = {
         'scene-0001', 'scene-0002', 'scene-0003', 'scene-0004', 'scene-0005',
         'scene-0006', 'scene-0007', 'scene-0008', 'scene-0009', 'scene-0010',
         'scene-0011', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
         'scene-0016', 'scene-0017', 'scene-0018', 'scene-0019', 'scene-0020',
         'scene-0021', 'scene-0022', 'scene-0023', 'scene-0024', 'scene-0025',
         'scene-0026', 'scene-0027', 'scene-0028', 'scene-0029', 'scene-0030',
         'scene-0031', 'scene-0032', 'scene-0033', 'scene-0034', 'scene-0035',
         'scene-0036', 'scene-0038', 'scene-0039', 'scene-0041', 'scene-0042',
         'scene-0043', 'scene-0044', 'scene-0045', 'scene-0046', 'scene-0047',
         'scene-0048', 'scene-0049', 'scene-0050', 'scene-0051', 'scene-0052',
         'scene-0053', 'scene-0054', 'scene-0055', 'scene-0056', 'scene-0057',
         'scene-0058', 'scene-0059', 'scene-0060', 'scene-0061', 'scene-0062',
         'scene-0063', 'scene-0064', 'scene-0065', 'scene-0066', 'scene-0067',
         'scene-0068', 'scene-0069', 'scene-0070', 'scene-0071', 'scene-0072',
         'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0092',
         'scene-0093', 'scene-0094', 'scene-0095', 'scene-0096', 'scene-0097',
         'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102'
     }

     split_data = []
     for split in ['train', 'val', 'test']:
          files = sorted(glob.glob(f'{data_root}/label/{split}/scene*.txt'))
          scenes = [os.path.splitext(os.path.basename(x))[0] for x in files]
          # Filter to only scenes in our 10% subset
          scenes = [s for s in scenes if s in subset_scenes]
          split_data.append(scenes)
     return split_data