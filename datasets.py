import torch
from torch.utils.data import Dataset, DataLoader
from trimesh.exchange.obj import load_obj
from trimesh import transformations
import numpy as np
import math
from itertools import product

class SingleModelData(Dataset):
    def __init__(self, fpath='data/torus.obj'):
        with open(fpath, 'rb') as f:
            obj = load_obj(f)
        self.vertices = obj['vertices']
        self.num_verts = self.vertices.shape[0]
        
    def __len__(self):
        # arbitrary large number so it doesn't complain
        return int(1e9)
        
    def __getitem__(self, idx):
        angle = np.random.uniform() * 2 * math.pi
        direction = random_direction()
        
        rot_matrix = transformations.rotation_matrix(angle, 
                                                     direction, 
                                                     [0, 0, 0])
        verts = np.column_stack([self.vertices, 
                                 np.ones((self.num_verts, 1))])
        verts = verts @ rot_matrix
        
        return torch.Tensor(verts[:, :3])
    
def random_direction():
    dirs = list(product([0, 1], repeat=3))
    dirs = [np.array(i).reshape(1, 3) for i in dirs]
    dirs = np.concatenate(dirs, axis=0)
    return dirs[np.random.randint(1, len(dirs)), :] # excludes [0, 0, 0]
        