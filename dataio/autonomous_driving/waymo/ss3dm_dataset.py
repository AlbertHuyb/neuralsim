"""
@file   waymo_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for Waymo Open Dataset - Perception v1
"""
import os
import sys
import pickle
import numpy as np
from glob import glob
from typing import Any, Dict, List
from scipy.spatial.transform import Rotation as R

from nr3d_lib.utils import load_rgb
from nr3d_lib.config import ConfigDict

from dataio.dataset_io import DatasetIO
from dataio.utils import clip_node_data, clip_node_segments
from .waymo_dataset import WaymoDataset

#---------------- Waymo original definition
WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
WAYMO_CAMERAS = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
WAYMO_LIDARS = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
# NUM_CAMERAS = len(WAYMO_CAMERAS)
# NUM_LIDARS = len(WAYMO_LIDARS)

#---------------- Cityscapes semantic segmentation
ss3dm_classes = [
    'None','Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole',
    'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain', 'Sky',
    'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle',
    'Bicycle','Static','Dynamic','Other','Water','RoadLines','Ground',
    'Bridge','RailTrack','GuardRail'
]
## change the first alphabet to lower case
ss3dm_classes = [cls[0].lower()+cls[1:] for cls in ss3dm_classes]
ss3dm_classes_ind_map = {cn: i for i, cn in enumerate(ss3dm_classes)}

cityscapes_dynamic_classes = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

cityscapes_human_classes = [
    'person', 'rider'
]

waymo_classes_in_cityscapes = {
    'unknwon': ['train'],
    'Vehicle': ['car', 'truck', 'bus'],
    'Pedestrian': ['person'],
    'Sign': ['traffic light', 'traffic sign'],
    'Cyclist': ['rider', 'motorcycle', 'bicycle']
}


#-----------------------------------------------------
#------- Dataset IMPL (implement standard APIs)
#-----------------------------------------------------
class SS3DMDataset(WaymoDataset):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config)

    def get_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.ones_like(raw).astype(np.bool8)
        ret[raw==ss3dm_classes_ind_map['sky']] = False
        # [H, W] 
        # Binary occupancy mask on RGB image. 1 for occpied, 0 for not.
        return ret.squeeze()
    def get_dynamic_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in cityscapes_dynamic_classes:
            ind = ss3dm_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for dynamic object, 0 for static.
        return ret.squeeze()
    def get_human_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in cityscapes_human_classes:
            ind = ss3dm_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for human-related object, 0 for other.
        return ret.squeeze()
    def get_road_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        ret[raw==ss3dm_classes_ind_map['road']] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for road semantics, 0 for other.
        return ret.squeeze()
    def get_semantic_mask_all(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.full(raw.shape, -1, dtype=np.int16)
        for waymo_ind, dataset_class_str in enumerate(WAYMO_CLASSES):
            for cls in waymo_classes_in_cityscapes[dataset_class_str]:
                ind = ss3dm_classes_ind_map[cls]
                ret[raw==ind] = waymo_ind
        # Integer semantic mask on RGB image.
        return ret.squeeze()
    def get_semantic_mask_of_class(self, scene_id: str, camera_id: str, frame_index: int, dataset_class_str: str, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in waymo_classes_in_cityscapes[dataset_class_str]:
            ind = ss3dm_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for selected class.
        return ret.squeeze()