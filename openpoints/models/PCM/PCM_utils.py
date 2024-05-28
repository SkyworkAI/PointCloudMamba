import torch
import torch.nn as nn
import torch.nn.functional as F
from .serialization import Point
import math

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, bias=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def serialization(pos, feat, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, feat, x_res


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    #dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    #dist += torch.sum(src ** 2, -1).view(B, N, 1)
    #dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    src_ = src.unsqueeze(2)
    dst_ = dst.unsqueeze(1)
    dist = (src_ - dst_) ** 2
    dist = dist.sum(dim=-1)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz, training=True):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    if training:
        sqrdists = square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    else:
        N = new_xyz.shape[1]
        idx_list = []
        n_splits = N // 1024
        if n_splits * 1024 != N:
            n_splits += 1
        start, end = 0, 1024
        for i in range(n_splits):
            end = min(end, N)
            sqrdists = square_distance(new_xyz[:, start: end], xyz)
            _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
            idx_list.append(group_idx)
            start += 1024
            end += 1024
        group_idx = torch.cat(idx_list, dim=1)
    return group_idx

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


