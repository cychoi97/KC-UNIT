import os
import random
import numpy as np

from tqdm import tqdm
from kmeans_pytorch import kmeans

import torch
import torch.nn.functional as F
import torchvision

from data_loader import get_loader
from model import (Generator_GGCL, Discriminator_GGCL)


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def create_labels(device, c_org, c_dim=3):
    """Generate target domain labels for debugging and testing."""
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def get_colors():
    dummy_color = np.array([
        [178, 34, 34],  # firebrick
        [0, 139, 139],  # dark cyan
        [245, 222, 179],  # wheat
        [25, 25, 112],  # midnight blue
        [255, 140, 0],  # dark orange
        [128, 128, 0],  # olive
        [50, 50, 50],  # dark grey
        [34, 139, 34],  # forest green
        [100, 149, 237],  # corn flower blue
        [153, 50, 204],  # dark orchid
        [240, 128, 128],  # light coral
    ])

    for t in (0.6, 0.3):  # just increase the number of colors for big K
        dummy_color = np.concatenate((dummy_color, dummy_color * t))

    dummy_color = (np.array(dummy_color) - 128.0) / 128.0
    dummy_color = torch.from_numpy(dummy_color)

    return dummy_color


def get_cluster_vis(device, feat, num_clusters=10, target_res=256):
    # feat : NCHW
    print('feature_size:', feat.size())
    img_num, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(img_num * H * W, -1)
    feat = feat.to(torch.float32).to(device)
    cluster_ids_x, cluster_centers = kmeans(
        X=feat, num_clusters=num_clusters, distance='cosine',
        tol=1e-4,
        device=device)

    cluster_ids_x = cluster_ids_x.to(device)
    cluster_centers = cluster_centers.to(device)
    color_rgb = get_colors().to(device)
    vis_img = []
    for idx in range(img_num):
        num_pixel = target_res * target_res
        current_res = cluster_ids_x[num_pixel * idx:num_pixel * (idx + 1)].to(device)
        color_ids = torch.index_select(color_rgb, 0, current_res)
        color_map = color_ids.permute(1, 0).view(1, 3, target_res, target_res)
        color_map = F.interpolate(color_map, size=(512, 512))
        vis_img.append(color_map.to(device))

    vis_img = torch.cat(vis_img, dim=0)

    return vis_img


# seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# dataloader config
data_path = '/workspace/changyong/01.CT_Standardization/data'
dataset = 'SIEMENS'
batch_size = 1
image_size = 512
mode = 'feat_vis'
num_workers=0

# load dataloader
dataloader = get_loader(batch_size, data_path, dataset=dataset, \
                        image_size=image_size, mode=mode, \
                        shuffle=False, num_workers=num_workers)

# model config
model_save_dir = '/workspace/changyong/01.CT_Standardization/GGCL/result/unpaired50/GGCL1_with_GGDR5_all_dec/models'
c_dim = 3
g_conv_dim = 64
g_repeat_num = 6
d_conv_dim = 32
d_repeat_num = 7

test_iters = [50000, 100000, 200000, 400000]

# visualize generator
result_dir = f'/workspace/changyong/01.CT_Standardization/GGCL/result/unpaired50/GGCL1_with_GGDR5_all_dec/results/feat_vis'
os.makedirs(result_dir, exist_ok=True)

G = Generator_GGCL(g_conv_dim, c_dim, g_repeat_num).to(device)
D = Discriminator_GGCL(image_size, d_conv_dim, c_dim, d_repeat_num).to(device)

for iters in test_iters:
    G_path = os.path.join(model_save_dir, f'{iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    print(f'Load {iters}-G.ckpt!')

    D_path = os.path.join(model_save_dir, f'{iters}-D.ckpt')
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    print(f'Load {iters}-D.ckpt!')
    
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(dataloader)):
            x_real = data_dict['image']
            c_org = data_dict['label']
            
            x_real = x_real.to(device)
            c_trg_list = create_labels(device, c_org, c_dim)
            # torch_ones = torch.ones(1,3,512,512).to(device)
            
            img_list = [] # torch.cat([x_real, x_real, x_real], dim=1)
            g_feat_list = []
            d_feat_list = []
            
            target_res = 256
            num_clusters = 12 # kmeans cluster
            
            for c_trg in c_trg_list:
                if c_trg[0][c_org[0]] != 1.: # except for i -> i
                    fake_img, _, g_feat = G(x_real, c_trg)
                    _, _, d_feat = D(fake_img)
                    g_vis_img = get_cluster_vis(device, g_feat, num_clusters=num_clusters, target_res=target_res)
                    d_vis_img = get_cluster_vis(device, d_feat, num_clusters=num_clusters, target_res=target_res)

                    img_list.append(torch.cat([fake_img, fake_img, fake_img], dim=1))
                    g_feat_list.append(g_vis_img)
                    d_feat_list.append(d_vis_img)
                
            for idx, val in enumerate(g_feat_list):
                g_feat_list[idx] = F.interpolate(val, size=(512, 512))
            for idx, val in enumerate(d_feat_list):
                d_feat_list[idx] = F.interpolate(val, size=(512, 512))
                
            img = torch.cat(img_list, dim=3)
            img = (img + 1) * 127.5 / 255.0
            g_feat = torch.cat(g_feat_list, dim=3)
            g_feat = (g_feat + 1) * 127.5 / 255.0
            d_feat = torch.cat(d_feat_list, dim=3)
            d_feat = (d_feat + 1) * 127.5 / 255.0
            
            result_path = os.path.join(result_dir, f'{i+1}_k-means_{num_clusters}_feature_from_{iters}iters_{str(c_org.numpy())}.png')
            
            vis_img = torch.cat([img, g_feat, d_feat], dim=2)
            torchvision.utils.save_image(vis_img, result_path)