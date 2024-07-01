import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=True
                                   )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class UEncoder(nn.Module):

    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)  # (112, 112, 64)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)  # (56, 56, 128)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)  # (28, 28, 256)

        x = self.res4(x)
        features.append(x)  # (28, 28, 512)
        x = self.pool4(x)  # (14, 14, 512)

        x = self.res5(x)
        features.append(x)  # (14, 14, 1024)
        x = self.pool5(x)  # (7, 7, 1024)
        features.append(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class HGNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super(HGNN, self).__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x, G):
        residual = x
        x = self.conv(x)
        x = G.matmul(x)
        x = F.relu(self.bn(x.permute(0,2,1).contiguous())).permute(0,2,1).contiguous() + residual
        return x


class HGNN_layer(nn.Module):
    """
        Writen by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, node = None, K_neigs=None, kernel_size=5, stride=2):
        super(HGNN_layer, self).__init__()
        self.HGNN = HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs

        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)

    def forward(self, x):


        B, N, C = x.shape
        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(x, k=self.K_neigs[0])
        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)
        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.
        Dv = Dv * alpha
        max_k = int(Dv.max())
        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(x, k=max_k - 1)
        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(x.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(x.device)
        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()
        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)
        local_H = self.local_H.repeat(B,1,1).to(new_H.device)

        _H = torch.cat([new_H,local_H],dim=2)
        _G = self._generate_G_from_H_b(_H)

        x = self.HGNN(x, _G)

        return x



    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape


        # the weight of the hyperedge

        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)
        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedge

        DE = torch.sum(H, dim=1)


        invDE = torch.diag_embed((torch.pow(DE, -1)))
        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)



        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G


    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            # Chai's weight function
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()

        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()

        H_local = torch.zeros((size * size, edge))


        H_local[inp_unf, matrix] = 1.

        return H_local


class HyperNet(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(HyperNet, self).__init__()
        self.HGNN_layer = HGNN_layer(channel, node = node, kernel_size=kernel_size, stride=stride, K_neigs=K_neigs)

    def forward(self, x):

        b,c,w,h = x.shape
        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        x = self.HGNN_layer(x)
        x = x.permute(0,2,1).contiguous().view(b,c,w,h)

        return x


class HyperEncoder(nn.Module):
    def __init__(self,channel = [512, 1024, 1024]):
        super(HyperEncoder, self).__init__()


        kernel_size  = 3
        stride = 1
        self.HGNN_layer1 = HyperNet(channel[0], node=28, kernel_size=kernel_size, stride=stride, K_neigs=[1])
        self.HGNN_layer2 = HyperNet(channel[1], node=14, kernel_size=kernel_size, stride=stride, K_neigs=[1])
        self.HGNN_layer3 = HyperNet(channel[2], node=7, kernel_size=kernel_size, stride=stride, K_neigs=[1])


    def forward(self, x):

        _, _, _, feature1, feature2, feature3 = x
        feature1 = self.HGNN_layer1(feature1)
        feature2 = self.HGNN_layer2(feature2)
        feature3 = self.HGNN_layer3(feature3)

        return [feature1,feature2,feature3]

class ParallEncoder(nn.Module):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()
        self.Encoder2 = HyperEncoder()
        self.fusion_module = nn.ModuleList()
        self.num_module = 3
        self.channel_list = [256, 512, 1024]
        self.fusion_list = [512, 1024, 1024]


        self.squeelayers = nn.ModuleList()
        for i in range(self.num_module):
            self.squeelayers.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], 1, 1)
            )

    def forward(self, x):
        skips = []
        features = self.Encoder1(x)

        feature_hyper = self.Encoder2(features)

        skips.extend(features[:3])
        for i in range(self.num_module):
            skip = self.squeelayers[i](torch.cat((feature_hyper[i], features[i + 3]), dim=1))
            skips.append(skip)

        return skips




class Model(nn.Module):
    def __init__(self, n_classes = 9):
        super(Model, self).__init__()
        self.p_encoder = ParallEncoder()
        self.encoder_channels = [1024, 512, 256, 128, 64]
        self.decoder1 = DecoderBlock(self.encoder_channels[0] + self.encoder_channels[0], self.encoder_channels[1])
        self.decoder2 = DecoderBlock(self.encoder_channels[1] + self.encoder_channels[1], self.encoder_channels[2])
        self.decoder3 = DecoderBlock(self.encoder_channels[2] + self.encoder_channels[2], self.encoder_channels[3])
        self.decoder4 = DecoderBlock(self.encoder_channels[3] + self.encoder_channels[3], self.encoder_channels[4])


        self.segmentation_head2 = SegmentationHead(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head3 = SegmentationHead(
            in_channels=128,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head4 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head5 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.decoder_final = DecoderBlock(in_channels=64, out_channels=64)


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_skips = self.p_encoder(x)

        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])
        x2_up = self.decoder2(x1_up, encoder_skips[-3])
        x3_up = self.decoder3(x2_up, encoder_skips[-4])
        x4_up = self.decoder4(x3_up, encoder_skips[-5])
        x_final = self.decoder_final(x4_up, None)


        x2_up = self.segmentation_head2(x2_up)
        x3_up = self.segmentation_head3(x3_up)
        x4_up = self.segmentation_head4(x4_up)
        logits = self.segmentation_head5(x_final)


        x2_up = F.interpolate(x2_up, scale_factor=8, mode='bilinear')
        x3_up = F.interpolate(x3_up, scale_factor=4, mode='bilinear')
        x4_up = F.interpolate(x4_up, scale_factor=2, mode='bilinear')




        return x2_up,x3_up,x4_up,logits

if __name__ == '__main__':

    model = Model(n_classes=9)
    inout = torch.randn((1, 1, 224, 224))
    x2_up,x3_up,x4_up,logits = model(inout)
    print( x2_up.shape,x3_up.shape,x4_up.shape,logits.shape)
    print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters())/1000000)

