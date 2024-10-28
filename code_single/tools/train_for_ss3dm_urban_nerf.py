### CUDA_VISIBLE_DEVICES=2 python code_single/tools/run.py train,eval,eval_lidar,extract_mesh --config code_single/configs/ss3dm/streetsurf/Town01_150_withmask_withlidar_withnormal_all_cameras.yaml --eval.downscale=2 --eval_lidar.lidar_id=lidar_TOP --extract_mesh.to_world --extract_mesh.res=0.1

import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--town_name', type=str, default=None)
parser.add_argument('--seq_name', type=str, default=None)
parser.add_argument('--mesh_res', type=float, default=0.1)
args = parser.parse_args() 

data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

if args.town_name is not None:
    town_list = [args.town_name]
else:
    town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']


for town_name in town_list:
    if os.path.isdir(os.path.join(data_root, town_name)):
        town_dir = os.path.join(data_root, town_name)
        
        if args.seq_name is not None:
            seq_list = [args.seq_name]
        else:
            seq_list = os.listdir(town_dir)
        
        # import pdb; pdb.set_trace()
        for seq_name in seq_list:
        # for seq_name in ['Town01_300']:
            if os.path.isdir(os.path.join(town_dir, seq_name)):
                for config_folder, config_name in zip(['urban_nerf'],['withmask_withlidar_withnormal_all_cameras']):
                    train_cmd = 'python code_single/tools/run.py train --config code_single/configs/ss3dm/{}/{}_{}.yaml'.format(config_folder, seq_name, config_name)
                    
                    ckpt_dir = os.path.join('/data/huyb/cvpr-2024/neuralsim/logs/ss3dm/urban_nerf/{}/{}/ckpts'.format(seq_name, config_name))
                    if not os.path.exists(os.path.join(ckpt_dir, 'final_00015000.pt')):
                        os.system(train_cmd)
                    
                    extract_mesh_cmd = 'python code_single/tools/run.py extract_mesh --config code_single/configs/ss3dm/{}/{}_{}.yaml --extract_mesh.to_world --extract_mesh.res={}'.format(config_folder, seq_name, config_name, args.mesh_res)
                    
                    save_mesh_dir = '/data/huyb/cvpr-2024/neuralsim/logs/ss3dm/urban_nerf/{}/{}/meshes'.format(seq_name, config_name)
                    existing_meshes = glob.glob(os.path.join(save_mesh_dir, '*.ply'))
                    if len(existing_meshes) == 0:
                        os.system(extract_mesh_cmd)
                
