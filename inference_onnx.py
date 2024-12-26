import torch
import torch.nn as nn
import numpy as np
import time

import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
from pointscope import PointScopeClient as PSC

import onnxruntime as ort
import open3d as o3d

def topk(data, num_topk):
    sort, idx = data.sort(descending=True)
    return sort[:num_topk], idx[:num_topk]

class SoftProcrustesLayer(nn.Module):
    def __init__(self, config):
        super(SoftProcrustesLayer, self).__init__()

        self.sample_rate = config.sample_rate
        self.max_condition_num = config.max_condition_num

    @staticmethod
    def batch_weighted_procrustes( X, Y, w, eps=0.0001):
        '''
        @param X: source frame [B, N,3]
        @param Y: target frame [B, N,3]
        @param w: weights [B, N,1]
        @param eps:
        @return:
        '''
        # https://ieeexplore.ieee.org/document/88573

        bsize = X.shape[0]
        device = X.device
        W1 = torch.abs(w).sum(dim=1, keepdim=True)
        w_norm = w / (W1 + eps)
        mean_X = (w_norm * X).sum(dim=1, keepdim=True)
        mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
        Sxy = torch.matmul( (Y - mean_Y).transpose(1,2), w_norm * (X - mean_X) )
        Sxy = Sxy.cpu().double()
        U, D, V = Sxy.svd() # small SVD runs faster on cpu
        condition = D.max(dim=1)[0] / D.min(dim=1)[0]
        S = torch.eye(3)[None].repeat(bsize,1,1).double()
        UV_det = U.det() * V.det()
        S[:, 2:3, 2:3] = UV_det.view(-1, 1,1)
        svT = torch.matmul( S, V.transpose(1,2) )
        R = torch.matmul( U, svT).float().to(device)
        t = mean_Y.transpose(1,2) - torch.matmul( R, mean_X.transpose(1,2) )
        return R, t, condition



    def forward(self,  conf_matrix,  src_pcd, tgt_pcd,  src_mask=None, tgt_mask=None):
        '''
        @param conf_matrix:
        @param src_pcd:
        @param tgt_pcd:
        @param src_mask:
        @param tgt_mask:
        @return:
        '''

        bsize, N, M = conf_matrix.shape

        # subsample correspondence
        if src_mask is None:
            src_len = torch.tensor([N])
        else:
            src_len = src_mask.sum(dim=1)
        if tgt_mask is None:
            tgt_len = torch.tensor([M])
        else:
            tgt_len = tgt_mask.sum(dim=1)

        entry_max, _ = torch.stack([src_len,tgt_len], dim=0).max(dim=0)
        entry_max = (entry_max * self.sample_rate).int()
        sample_n_points = entry_max.float().mean().int() #entry_max.max()
        conf, idx = conf_matrix.view(bsize, -1).sort(descending=True,dim=1)
        w = conf [:, :sample_n_points]
        idx= idx[:, :sample_n_points]
        idx_src = idx//M #torch.div(idx, M, rounding_mode='trunc')
        idx_tgt = idx%M
        b_index = torch.arange(bsize).view(-1, 1).repeat((1, sample_n_points)).view(-1)
        src_pcd_sampled = src_pcd[b_index, idx_src.view(-1)].view(bsize, sample_n_points, -1)
        tgt_pcd_sampled = tgt_pcd[b_index, idx_tgt.view(-1)].view(bsize, sample_n_points, -1)
        w_mask = torch.arange(sample_n_points).view(1,-1).repeat(bsize,1).to(w)
        w_mask = w_mask < entry_max[:,None]
        w[~w_mask] = 0.

        # solve
        try :
            R, t, condition = self.batch_weighted_procrustes(src_pcd_sampled, tgt_pcd_sampled, w[...,None])
        except: # fail to get valid solution, this usually happens at the early stage of training
            R = torch.eye(3)[None].repeat(bsize,1,1).type_as(conf_matrix)
            t = torch.zeros(3, 1)[None].repeat(bsize,1,1).type_as(conf_matrix)
            condition = torch.zeros(bsize).type_as(conf_matrix)

        #filter unreliable solution with condition nnumber
        solution_mask = condition < self.max_condition_num
        R_forwd = R.clone()
        t_forwd = t.clone()
        R_forwd[~solution_mask] = torch.eye(3).type_as(R)
        t_forwd[~solution_mask] = torch.zeros(3, 1).type_as(R)

        return R, t, R_forwd, t_forwd, condition, solution_mask


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def hierarchical_downsample_pcd(batched_points, batched_lengths, config):
    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    # construt kpfcn inds
    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            config.neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            config.neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          config.neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    # coarse infomation
    coarse_level = config.coarse_level
    pts_num_coarse = input_batches_len[coarse_level].view(-1, 2)
    src_ind_coarse = []
    tgt_ind_coarse = []
    accumu = 0

    for cnt in pts_num_coarse: #input_batches_len[-1].numpy().reshape(-1,2)) :

        n_s_pts, n_t_pts = cnt

        '''split indices of bottleneck feats'''
        src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
        tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )

        accumu = accumu + n_s_pts + n_t_pts

    src_ind_coarse = torch.cat(src_ind_coarse)
    tgt_ind_coarse = torch.cat(tgt_ind_coarse)

    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'stack_lengths': input_batches_len,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
    }

    return dict_inputs


class Pipeline(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ort.InferenceSession('onnx_export/backbone.onnx')
        self.coarse_transformer = ort.InferenceSession(
            'onnx_export/transformer.onnx', 
        )
        self.max_points = 15000
        self.rt_method = "RANSAC"
        # self.rt_method = "SoftProcrustes"
        if self.rt_method == "SoftProcrustes":
            self.coarse_matching = ort.InferenceSession('onnx_export/matching.onnx')
            self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])

    def preprocess_data(self, src_pcd, tgt_pcd, config):
        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        batched_points = np.concatenate([src_pcd, tgt_pcd], axis=0)
        batched_features = np.concatenate([src_feats, tgt_feats], axis=0)

        batched_features = torch.from_numpy(batched_features)
        batched_points = torch.from_numpy(batched_points)
        batched_lengths = torch.tensor([src_pcd.shape[0], tgt_pcd.shape[0]]).long()

        data = hierarchical_downsample_pcd(batched_points, batched_lengths, config.kpfcn_config)
        
        data['src_pcd_list'] = [src_pcd]
        data['tgt_pcd_list'] = [tgt_pcd]
        data['features'] = batched_features

        return data


    def forward(self, data):

        data = self.preprocess_data(data['src'], data['tgt'], self.config)

        inp = {f'points_{ind}': i.numpy() for ind, i in enumerate(data['points'])}
        inp.update({f'neighbors_{ind}': i.numpy() for ind, i in enumerate(data['neighbors'])})
        inp.update({'features': data['features'].numpy()})
        inp.update({f'pools_{ind}': i.numpy() for ind, i in enumerate(data['pools'][:3])})
        inp.update({f'upsamples_{2}':data['upsamples'][2].numpy()})
        coarse_feats = self.backbone.run(None, inp)

        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]
        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']
        src_feats = coarse_feats[0][src_ind_coarse]
        tgt_feats = coarse_feats[0][tgt_ind_coarse]
        s_pcd = pcd[src_ind_coarse].numpy()
        t_pcd = pcd[tgt_ind_coarse].numpy()

        point_num_diff = s_pcd.shape[0] - t_pcd.shape[0]
        if point_num_diff > 0:
            t_pcd = np.concatenate([t_pcd, np.zeros((point_num_diff, t_pcd.shape[-1]))]).astype(np.float32)
            tgt_feats = np.concatenate([tgt_feats, np.zeros((point_num_diff, tgt_feats.shape[-1]))]).astype(np.float32)
        elif point_num_diff < 0:
            s_pcd = np.concatenate([s_pcd, np.zeros((-point_num_diff, s_pcd.shape[-1]))]).astype(np.float32)
            src_feats = np.concatenate([src_feats, np.zeros((-point_num_diff, src_feats.shape[-1]))]).astype(np.float32)

        src_feats_cond, tgt_feats_cond, src_pe, tgt_pe = self.coarse_transformer.run(None, {
            "src_feats": src_feats[None, ...],
            "tgt_feats": tgt_feats[None, ...],
            "s_pcd": s_pcd[None, ...],
            "t_pcd": t_pcd[None, ...],
        })

        if point_num_diff > 0:
            t_pcd = t_pcd[:-point_num_diff]
        elif point_num_diff < 0:
            s_pcd = s_pcd[:point_num_diff]

        pred = self.get_transformation(s_pcd, t_pcd, src_feats_cond, tgt_feats_cond, src_pe, tgt_pe)
        data["pred"] = pred
        return data
    
    def get_transformation(self, s_pcd, t_pcd, src_feats_cond, tgt_feats_cond, src_pe, tgt_pe):
        if self.rt_method == "RANSAC":
            src_pcd = to_o3d_pcd(s_pcd)
            tgt_pcd = to_o3d_pcd(t_pcd)
            src_feats = to_o3d_feats(src_feats_cond[0, : s_pcd.shape[0]])
            tgt_feats = to_o3d_feats(tgt_feats_cond[0, : t_pcd.shape[0]])
            distance_threshold = 0.05
            ransac_n = 3
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_pcd, tgt_pcd, src_feats, tgt_feats, False, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
            R = result_ransac.transformation[:3, :3]
            t = result_ransac.transformation[:3, 3:4]
        elif self.rt_method == "SoftProcrustes":
            conf_matrix_pred = self.coarse_matching.run(None, {
                "src_feats_cond": src_feats_cond,
                "tgt_feats_cond": tgt_feats_cond,
                "src_pe": src_pe,
                "tgt_pe": tgt_pe,
            })
            conf_matrix_pred = conf_matrix_pred[0]
            conf_matrix_pred = conf_matrix_pred[:, :s_pcd.shape[0], :t_pcd.shape[0]]
            R, t, _, _, _, _ = self.soft_procrustes(
                torch.from_numpy(conf_matrix_pred).contiguous(), 
                torch.from_numpy(s_pcd)[None, ...], 
                torch.from_numpy(t_pcd)[None, ...])
        pred = np.eye(4)
        pred[:3, :3] = R
        pred[:3, -1:] = t
        return pred

    def evaluate(self, pred, gt):
        src_gt = gt["src_gt"]
        tgt_gt = gt["tgt_gt"]
    
        # Visualize the predicted transformation
        PSC().add_pcd(src, pred).add_pcd(tgt).show()

        rot, trans = pred[..., :3, :3].numpy(), pred[..., :3, 3:4].numpy()
        src_pred = src @ rot.T + trans.T
        PSC().add_pcd(src_pred[:, :-1]).add_pcd(tgt).show()
        
        src_ = torch.from_numpy(src).float()
        src_h = torch.cat([src_, torch.ones_like(src_[:, :1])], dim=1)
        src_pred = src_h @ pred.T
        PSC().add_pcd(src_pred[:, :-1]).add_pcd(tgt).show()

        src_ = torch.from_numpy(src).float()
        # src_ = src.clone()
        src_h = torch.cat([src_, torch.ones_like(src_[:, :1])], dim=1)
        src_gt_trans = src_h @ src_gt.T @ tgt_gt.T.inverse()
        PSC().add_pcd(src_gt_trans[:, :-1]).add_pcd(tgt).show()

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

def load_pcd(model_path):
    if model_path.endswith(".pth"):
        return torch.from_numpy(torch.load(model_path)).float()
    elif model_path.endswith(".ply"):
        return np.asarray(o3d.io.read_point_cloud(model_path).points) / 10

if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    from configs.models import architectures
    import os
    yaml.add_constructor('!join', join)
    config_name = "configs/test/3dmatch.yaml"

    with open(config_name,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)
    config.kpfcn_config.architecture = architectures[config.dataset]
    
    pipeline = Pipeline(config)

    # scene = "7-scenes-redkitchen"
    scene = "sun3d-hotel_uc-scan3"
    root_dir = "/home/sheldon/sharefd/data/indoor/test"
    src_file = os.path.join(root_dir, scene, "cloud_bin_0.pth")
    tgt_file = os.path.join(root_dir, scene, "cloud_bin_2.pth")

    # root_dir = "/home/sheldon/sharefd/data/data/"
    # src_file = os.path.join(root_dir, "oriMesh1_1.ply")
    # tgt_file = os.path.join(root_dir, "oriMesh3_1.ply")

    src = load_pcd(src_file)
    tgt = load_pcd(tgt_file)

    start = time.time()
    res = pipeline({
        "src": src, 
        "tgt": tgt
    })
    print("Total time: {}ms".format((time.time() - start) * 1000))

    src_gt_file = src_file.replace("pth", "info.txt")
    tgt_gt_file = tgt_file.replace("pth", "info.txt")
    if os.path.exists(src_gt_file) and os.path.exists(tgt_gt_file):
        with open(src_gt_file, "r") as f:
            src_gt = f.read()
            src_gt = torch.tensor([[float(x.strip()) for x in i.strip().split("\t")] for i in src_gt.strip().split("\n")[1:]])
        with open(tgt_gt_file, "r") as f:
            tgt_gt = f.read()
            tgt_gt = torch.tensor([[float(x.strip()) for x in i.strip().split("\t")] for i in tgt_gt.strip().split("\n")[1:]])
        print(src_gt)
        print(tgt_gt)

        pipeline.evaluate(res["pred"], {
            "src_gt": src_gt,
            "tgt_gt": tgt_gt
        })

