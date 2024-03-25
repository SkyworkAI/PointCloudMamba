from ..layers import furthest_point_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from .PCM_utils import knn_point, index_points, square_distance

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

class LocalGrouper(nn.Module):
    def __init__(self, channel, sample_ratio, kneighbors=12, use_xyz=True, normalize="center", k_stride=1, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.sample_ratio = sample_ratio
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.k_stride = k_stride
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points, points_res):
        B, N, C = xyz.shape
        S = N // self.sample_ratio
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        if S == N:
            new_xyz = xyz  # [B, npoint, 3]
            new_points = points  # [B, npoint, d]
        else:
            fps_idx = furthest_point_sample(xyz, S).long()  # [B, npoint]
            fps_idx = torch.sort(fps_idx, dim=-1)[0]
            new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
            new_points = index_points(points, fps_idx)  # [B, npoint, d]
            if points_res is not None:
                points_res = index_points(points_res, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz, training=self.training)
        idx = idx[:, :, ::self.k_stride]
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors // self.k_stride, 1)], dim=-1)
        return new_xyz, new_points, points_res

class LocalGrouper_withoutKNN(nn.Module):
    def __init__(self, channel, sample_ratio, use_xyz=True, normalize="center", **kwargs):
        """
        only down sample
        """
        super(LocalGrouper_withoutKNN, self).__init__()
        self.sample_ratio = sample_ratio
        self.use_xyz = use_xyz

    def forward(self, xyz, points, points_res):
        B, N, C = xyz.shape
        S = N // self.sample_ratio
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        if S == N:
            new_xyz = xyz  # [B, npoint, 3]
            new_points = points  # [B, npoint, d]
        else:
            fps_idx = furthest_point_sample(xyz, S).long()  # [B, npoint]
            fps_idx = torch.sort(fps_idx, dim=-1)[0]
            new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
            new_points = index_points(points, fps_idx)  # [B, npoint, d]
            if points_res is not None:
                points_res = index_points(points_res, fps_idx)
        if self.use_xyz:
            new_points = torch.cat([new_xyz, new_points], dim=-1)
        return new_xyz, new_points.unsqueeze(2), points_res

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True, knn_grouper=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        if knn_grouper:
            in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        else:
            in_channels = 3 + channels if use_xyz else channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class PreExtraction_Replace(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction_Replace, self).__init__()
        # only for max pool1d

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, _ = x.size()
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        if blocks == 0:
            self.extraction = nn.Identity()
        else:
            self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                            res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2, k=3):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            if self.training:
                dists = square_distance(xyz1, xyz2)
                dists, idx = dists.sort(dim=-1)
                dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            else:
                dists_list = []
                idx_list = []
                n_splits = N // 1024
                if n_splits * 1024 != N:
                    n_splits += 1
                start, end = 0, 1024
                for i in range(n_splits):
                    end = min(end, N)
                    dists = square_distance(xyz1[:, start: end], xyz2)
                    dists, idx = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
                    dists_list.append(dists)
                    idx_list.append(idx)
                    start += 1024
                    end += 1024
                dists = torch.cat(dists_list, dim=1)
                idx = torch.cat(idx_list, dim=1)

            # dists, idx = dists.sort(dim=-1)
            # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # print(points2.shape, '  ', idx.shape)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points
