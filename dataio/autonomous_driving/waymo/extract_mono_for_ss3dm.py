# python extract_mono_cues.py --task=depth --data_root=/data/huyb/data/streetsurf_data/ --omnidata_path=/data/huyb/hyb_106/reconstruction/omnidata/omnidata_tools/torch/

import os

data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

# for town_name in os.listdir(data_root):
for town_name in ['Town10']:
    if os.path.isdir(os.path.join(data_root, town_name)):
        town_dir = os.path.join(data_root, town_name)
                
        cmd = 'python extract_mono_cues.py --task=depth --data_root={} --omnidata_path=/data/huyb/hyb_106/reconstruction/omnidata/omnidata_tools/torch/'.format(town_dir)
        os.system(cmd)
        cmd = 'python extract_mono_cues.py --task=normal --data_root={} --omnidata_path=/data/huyb/hyb_106/reconstruction/omnidata/omnidata_tools/torch/'.format(town_dir)
        os.system(cmd)