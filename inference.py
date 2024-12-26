from models.blocks import *
from models.transformer import RepositioningTransformer
from models.matching import Matching
from models.procrustes import SoftProcrustesLayer
# from lib.tictok import Timers
import numpy as np

import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
from pointscope import PointScopeClient as PSC

import open3d as o3d


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
    b_size = pts_num_coarse.shape[0]
    src_pts_max, tgt_pts_max = pts_num_coarse.amax(dim=0)
    coarse_pcd = input_points[coarse_level] # .numpy()
    coarse_matches= []
    src_ind_coarse_split= [] # src_feats shape :[b_size * src_pts_max]
    src_ind_coarse = []
    tgt_ind_coarse_split= []
    tgt_ind_coarse = []
    accumu = 0
    src_mask = torch.zeros([b_size, src_pts_max], dtype=torch.bool)
    tgt_mask = torch.zeros([b_size, tgt_pts_max], dtype=torch.bool)

    #grid subsample fine level points for differentiable matching
    fine_pts, fine_length = batch_grid_subsampling_kpconv(input_points[0], input_batches_len[0], sampleDl=dl*0.5*0.85)
    fine_ind = batch_neighbors_kpconv(fine_pts, input_points[0], fine_length, input_batches_len[0], dl*0.5*0.85, 1).squeeze().long()

    for entry_id, cnt in enumerate( pts_num_coarse ): #input_batches_len[-1].numpy().reshape(-1,2)) :

        n_s_pts, n_t_pts = cnt

        '''split mask for bottlenect feats'''
        src_mask[entry_id][:n_s_pts] = 1
        tgt_mask[entry_id][:n_t_pts] = 1

        '''split indices of bottleneck feats'''
        src_ind_coarse_split.append( torch.arange( n_s_pts ) + entry_id * src_pts_max )
        tgt_ind_coarse_split.append( torch.arange( n_t_pts ) + entry_id * tgt_pts_max )
        src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
        tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )

        '''get match at coarse level'''
        c_src_pcd = coarse_pcd[accumu : accumu + n_s_pts]
        c_tgt_pcd = coarse_pcd[accumu + n_s_pts: accumu + n_s_pts + n_t_pts]
        # s_pc_wrapped = (torch.matmul( batched_rot[entry_id], c_src_pcd.T ) + batched_trn [entry_id]).T
        # coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped.numpy(), c_tgt_pcd.numpy(), search_radius=config['coarse_match_radius'])  )# 0.1m scaled
        # coarse_matches.append(coarse_match_gt)

        accumu = accumu + n_s_pts + n_t_pts

    src_ind_coarse_split = torch.cat(src_ind_coarse_split)
    tgt_ind_coarse_split = torch.cat(tgt_ind_coarse_split)
    src_ind_coarse = torch.cat(src_ind_coarse)
    tgt_ind_coarse = torch.cat(tgt_ind_coarse)

    dict_inputs = {

        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'stack_lengths': input_batches_len,
        'coarse_matches': coarse_matches,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_ind_coarse_split': src_ind_coarse_split,
        'tgt_ind_coarse_split': tgt_ind_coarse_split,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
        # 'batched_rot': batched_rot,
        # 'batched_trn': batched_trn,
        # 'gt_cov': gt_cov_list,
        #for refine
        # 'correspondences_list': correspondences_list,
        # 'fine_ind': fine_ind,
        # 'fine_pts': fine_pts,
        # 'fine_length': fine_length
    }

    return dict_inputs



class KPFCN(nn.Module):

    def __init__(self, config):
        super(KPFCN, self).__init__()

        ############
        # Parameters
        ############
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim

        #####################
        # List Encoder blocks
        #####################
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # bottleneck output & input layer

        self.coarse_out = nn.Conv1d(in_dim//2, config.coarse_feature_dim,  kernel_size=1, bias=True)
        coarse_in_dim = config.coarse_feature_dim
        self.coarse_in = nn.Conv1d(coarse_in_dim, in_dim//2,  kernel_size=1, bias=True)

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2


        #####################
        # fine output layer
        #####################
        fine_feature_dim =  config.fine_feature_dim
        self.fine_out = nn.Conv1d(out_dim, fine_feature_dim, kernel_size=1, bias=True)

    def forward(self, batch):
        # Get input features

        x = batch['features'].clone().detach()
        # 1. joint encoder part
        self.skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                self.skip_x.append(x)
            x = block_op(x, batch)  # [N,C]
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, self.skip_x.pop()], dim=1)
            x = block_op(x, batch)
            if block_i == 1 :
                coarse_feats = x.transpose(0,1).unsqueeze(0)  #[B, C, N]
                coarse_feats = self.coarse_out(coarse_feats)  #[B, C, N]
                coarse_feats = coarse_feats.transpose(1,2).squeeze(0)

                return coarse_feats #[N,C2]


class Pipeline(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = KPFCN(config['kpfcn_config'])
        self.pe_type = config['coarse_transformer']['pe_type']
        self.positioning_type = config['coarse_transformer']['positioning_type']
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])
        self.max_points = 15000

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


    def forward(self, data,  timers=None, export=False):

        self.timers = timers
        if export:
            export_root = "onnx_export"
            os.makedirs(export_root, exist_ok=True)

        data = self.preprocess_data(data['src'], data['tgt'], self.config)

        if self.timers: self.timers.tic('kpfcn backbone encode')
        key_names = ['points', 'neighbors', 'features', 'pools', 'upsamples']
        backbone_data = {i:data[i] for i in key_names}
        coarse_feats = self.backbone(backbone_data)
        if self.timers: self.timers.toc('kpfcn backbone encode')
        if export:
            export_path = os.path.join(export_root, "backbone.onnx")
            input_names = [f"points_{i}" for i in range(len(data['points']))] + \
                          [f"neighbors_{i}" for i in range(len(data['neighbors']))] + \
                          ["features"] + \
                          [f"pools_{i}" for i in range(len(data['pools']))] + \
                          [f"upsamples_{i}" for i in range(len(data['upsamples']))]
            output_names = ["coarse_feats"]
            dynamic_axes={name: {0: 'N'} for name in input_names+output_names}
            for i in dynamic_axes.keys():
                if i.split('_')[0] in ['neighbors', 'pools', 'upsamples']:
                    dynamic_axes[i].update({1: 'P'})
            self.export(
                self.backbone, backbone_data, 
                input_names, output_names, export_path,
                dynamic_axes=dynamic_axes
            )

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats(coarse_feats, data)
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })
        if self.timers: self.timers.toc('coarse_preprocess')

        if self.timers: self.timers.tic('coarse feature transformer')
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, timers=timers)
        if self.timers: self.timers.toc('coarse feature transformer')
        if export:
            export_path = os.path.join(export_root, "transformer.onnx")
            self.coarse_transformer.forward = self.coarse_transformer._forward_packed
            input_names = ["src_feats", "tgt_feats", "s_pcd", "t_pcd"]
            output_names = ["src_feats_cond", "tgt_feats_cond", "src_pe", "tgt_pe"]
            self.export(
                self.coarse_transformer, [src_feats, tgt_feats, s_pcd, t_pcd], 
                input_names, output_names, export_path
            )
            self.coarse_transformer.forward = self.coarse_transformer._forward

        if self.timers: self.timers.tic('match feature coarse')
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')
        if export:
            export_path = os.path.join(export_root, "matching.onnx")
            self.coarse_matching.forward = self.coarse_matching._forward_packed
            input_names = ["src_feats_cond", "tgt_feats_cond", "src_pe", "tgt_pe"]
            output_names = ["conf_matrix_pred"]
            dynamic_axes={name: {1: 'N'} for name in input_names+output_names}
            dynamic_axes[output_names[0]] = {1: 'S', 2: 'T'}
            self.export(
                self.coarse_matching, [src_feats, tgt_feats, src_pe, tgt_pe], 
                input_names, output_names, export_path,
                dynamic_axes=dynamic_axes
            )
            self.coarse_matching.forward = self.coarse_matching._forward

        if self.timers: self.timers.tic('procrustes_layer')
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')

        return data


    def split_feats(self, geo_feats, data):

        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        src_ind_coarse_split = data[ 'src_ind_coarse_split']
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask

    def export(self, model, inputs, input_names, output_names, path, dynamic_axe=1, dynamic_axes=None):
        if not dynamic_axes:
            dynamic_axes = {name: {dynamic_axe: 'N'} for name in input_names+output_names}
        torch.onnx.export(
            model,                      # model being run
            inputs,                     # model input (or a tuple for multiple inputs)
            path,                       # where to save the model (can be a file or file-like object)
            export_params=True,         # store the trained parameter weights inside the model file
            opset_version=12,           # the ONNX version to export the model to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=input_names,    # the model's input names
            output_names=output_names,  # the model's output names
            dynamic_axes=dynamic_axes,  # the model's dynamic axes
        )

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
        return np.asarray(o3d.io.read_point_cloud(model_path).points)/10

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
    
    model_path = 'pretrained/3dmatch/model_best_loss.pth'
    pipeline = Pipeline(config)
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"]
    pipeline.load_state_dict(model_state_dict)
    pipeline.eval()

    root_dir = "/home/sheldon/sharefd/data/indoor/test/7-scenes-redkitchen"
    src_file = os.path.join(root_dir, "cloud_bin_0.pth")
    tgt_file = os.path.join(root_dir, "cloud_bin_1.pth")

    # root_dir = "/home/sheldon/sharefd/data/data/"
    # src_file = os.path.join(root_dir, "oriMesh1_1.ply")
    # tgt_file = os.path.join(root_dir, "oriMesh3_1.ply")

    src = load_pcd(src_file)
    tgt = load_pcd(tgt_file)
    # ts = Timers()
    res = pipeline({
        "src": src, 
        "tgt": tgt
    }, export=True)
    # ts.print()

    src_gt_file = src_file.replace("pth", "info.txt")
    tgt_gt_file = tgt_file.replace("pth", "info.txt")
    with open(src_gt_file, "r") as f:
        src_gt = f.read()
        src_gt = torch.tensor([[float(x.strip()) for x in i.strip().split("\t")] for i in src_gt.strip().split("\n")[1:]])
    with open(tgt_gt_file, "r") as f:
        tgt_gt = f.read()
        tgt_gt = torch.tensor([[float(x.strip()) for x in i.strip().split("\t")] for i in tgt_gt.strip().split("\n")[1:]])
    print(src_gt)
    print(tgt_gt)
    pred = torch.eye(4)
    pred[:3, :3] = res["R_s2t_pred"].detach()
    pred[:3, -1:] = res["t_s2t_pred"].detach()

    pipeline.evaluate(pred, {
        "src_gt": src_gt,
        "tgt_gt": tgt_gt
    })

