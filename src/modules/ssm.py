import torch
import torch.nn.functional as F
from torch import nn

from modules import resnet


class CorrCosine(nn.Module):
    def __init__(self):
        super().__init__()
        self.corr_conv = CorrConv()

    def forward(self, ref_features, cur_features):
        ref_features = F.normalize(ref_features, p=2, dim=1)
        cur_features = F.normalize(cur_features, p=2, dim=1)
        sim_matrix = self.corr_conv(ref_features, cur_features)
        return sim_matrix


class CorrConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ref_features, cur_features):
        batch_size, num_channels, ref_h, ref_w = ref_features.shape
        _, _, cur_h, cur_w = cur_features.shape
        cur_features = cur_features.permute(0, 2, 3, 1).contiguous().view(
            batch_size * cur_h * cur_w, num_channels, 1, 1)
        ref_features = ref_features.view(1, batch_size * num_channels, ref_h, ref_w)
        corr_features = F.conv2d(ref_features, cur_features, groups=batch_size)
        corr_features = corr_features.view(batch_size, cur_h, cur_w, ref_h, ref_w)

        return corr_features


class SelfStructure(nn.Module):
    def __init__(self, keep_topk):
        super().__init__()
        self.keep_topk = keep_topk
        self.group_conv = nn.Sequential(
            nn.Conv2d(1024, 1024, (1, 1), groups=keep_topk),
            nn.ReLU(inplace=True)
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(1024, 1024 // keep_topk, (1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, corr_features, cur_features, ref_mask):
        # top k pixels in current frame that match one pixel in reference frame
        batch_size, cur_h, cur_w, ref_h, ref_w = corr_features.shape
        _, channels, _, _ = cur_features.shape
        cur_features = cur_features.view(batch_size, channels, cur_h * cur_w)

        corr_features = corr_features.view(batch_size, cur_h * cur_w, ref_h * ref_w)

        ref_mask = F.interpolate(ref_mask, (ref_h, ref_w), mode='bilinear', align_corners=False)
        ref_mask = ref_mask.view(batch_size, 1, ref_h * ref_w)
        fg_corr = corr_features * (ref_mask > 0.5).type(torch.float32)

        fg_struct = self.get_struct_info(fg_corr, cur_features)
        struct_info = fg_struct

        struct_info = torch.bmm(struct_info, cur_features)
        num_points = struct_info.shape[1]
        struct_info = struct_info.view(batch_size, num_points, cur_h, cur_w)

        # group conv
        cur_features = cur_features.view(batch_size, channels, cur_h, cur_w)
        group_features = self.group_conv(cur_features)
        group_struct = struct_info.repeat(1, 1, channels // num_points, 1).view(batch_size, channels, cur_h,
                                                                                cur_w)
        group_struct *= group_features
        group_struct = group_struct.view(batch_size, num_points, channels // num_points, cur_h, cur_w).sum(1)

        global_features = self.global_conv(cur_features)
        global_struct = struct_info.mean(dim=1, keepdim=True)
        global_struct = global_struct * global_features

        fusion = torch.cat([group_struct, global_struct], dim=1)
        return fusion

    def get_struct_info(self, corr, cur_features):
        batch_size = cur_features.shape[0]
        corr = torch.sum(corr, dim=-1)
        _, topk_indices = torch.topk(corr, self.keep_topk, dim=1)
        struct_info = cur_features[[[b] for b in range(batch_size)], :, topk_indices]  # [b, k, c]
        return struct_info


class PixelSimilarity(nn.Module):
    def __init__(self, matching, keep_topk):
        super().__init__()
        self.out_channels = 2 * keep_topk
        self.out_channels += 1024 // keep_topk * 2
        self.soft_matching = {'conv': CorrConv, 'cosine': CorrCosine}[matching]()
        self.mask_topk = MaskedTopk(keep_topk)
        self.self_structure = SelfStructure(keep_topk)
        self.conv1x1 = nn.Conv2d(1024 // keep_topk * 2, 2, (1, 1))

    def forward(self, ref_features, cur_features, ref_mask):
        corr_features = self.soft_matching(ref_features, cur_features)
        pixel_corr = self.mask_topk(corr_features, ref_mask)
        if self.self_structure:
            struct_corr = self.self_structure(corr_features, ref_features, ref_mask)
            pixel_corr = torch.cat([pixel_corr, struct_corr], dim=1)
        return pixel_corr, self.conv1x1(struct_corr)


class MaskedTopk(nn.Module):
    def __init__(self, keep_topk):
        super().__init__()
        self.keep_topk = keep_topk

    def forward(self, corr_features, ref_mask):
        batch_size, cur_h, cur_w, ref_h, ref_w = corr_features.shape
        corr_features = corr_features.view(batch_size, cur_h * cur_w, ref_h * ref_w)
        ref_mask = F.interpolate(ref_mask, (ref_h, ref_w), mode='bilinear', align_corners=False)
        ref_mask = ref_mask.view(batch_size, 1, ref_h * ref_w)

        fg_corr = corr_features * (ref_mask > 0.5).type(torch.float32)
        bg_corr = corr_features * (ref_mask <= 0.5).type(torch.float32)

        fg_corr, _ = torch.topk(fg_corr, self.keep_topk, dim=-1)
        bg_corr, _ = torch.topk(bg_corr, self.keep_topk, dim=-1)

        fg_corr = fg_corr.view(batch_size, cur_h, cur_w, self.keep_topk).permute(0, 3, 1, 2)
        bg_corr = bg_corr.view(batch_size, cur_h, cur_w, self.keep_topk).permute(0, 3, 1, 2)

        corr_map = torch.cat([bg_corr, fg_corr], dim=1)

        return corr_map


class ResidualBlock(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.res = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(v, v, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(v, v, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        x = x + self.res(x)
        return x


class GlobalConvolutionBlock(nn.Module):
    def __init__(self, in_dim, out_dim=256, k=7):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, k), padding=(0, k // 2), bias=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(k, 1), padding=(0, k // 2), bias=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, k), padding=(k // 2, 0), bias=True)
        )

        self.RB = ResidualBlock(out_dim)

    def forward(self, x):
        out = self.branch1(x) + self.branch2(x)
        out = self.RB(out)
        return out


class RefinementModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=True)
        self.RB1 = ResidualBlock(out_dim)
        self.RB2 = ResidualBlock(out_dim)

    def forward(self, x_top, x_low):
        _, _, h, w = x_low.size()

        x_top = F.interpolate(x_top, size=(h, w), mode='bilinear', align_corners=False)
        x_low = self.RB1(self.conv(x_low))
        x = x_top + x_low
        x = self.RB2(x)
        return x


class SSM(nn.Module):
    def __init__(self, encoder, matching, keep_topk):
        super().__init__()

        Encoder = {'resnet50': resnet.resnet50, 'resnet101': resnet.resnet101}[encoder]
        self.encoder = Encoder(pretrained=True)

        self.pixel_corr = PixelSimilarity(matching, keep_topk)

        self.GCB = GlobalConvolutionBlock(4096)
        self.RM1 = RefinementModule(1024 + self.pixel_corr.out_channels, 256)
        self.RM2 = RefinementModule(512 + self.pixel_corr.out_channels, 256)
        self.RM3 = RefinementModule(256 + self.pixel_corr.out_channels, 256)

        classifier_in_channels = 256
        self.classifier = nn.Conv2d(classifier_in_channels, 2, kernel_size=3, padding=1)
        self.dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, ref_img, ref_mask, cur_img, prev_mask):
        _, _, cur_h, cur_w = cur_img.shape

        ref_img_mask = torch.cat([ref_img, ref_mask], dim=1)
        ref_x5, ref_x4, _, _, _ = self.encoder(ref_img_mask)

        cur_img_mask = torch.cat([cur_img, prev_mask], dim=1)
        cur_x5, cur_x4, cur_x3, cur_x2, _ = self.encoder(cur_img_mask)

        ref_x5 = F.interpolate(ref_x5, cur_x5.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([ref_x5, cur_x5], dim=1)

        x = self.GCB(x)

        pixel_map, struct_mask = self.pixel_corr(ref_x4, cur_x4, ref_mask)
        corr_maps = pixel_map

        cur_x4 = torch.cat([cur_x4, corr_maps], dim=1)

        corr_maps = F.interpolate(corr_maps, cur_x3.shape[-2:], mode='nearest')
        cur_x3 = torch.cat([cur_x3, corr_maps], dim=1)

        corr_maps = F.interpolate(corr_maps, cur_x2.shape[-2:], mode='nearest')
        cur_x2 = torch.cat([cur_x2, corr_maps], dim=1)

        x = self.RM1(x, cur_x4)
        x = self.RM2(x, cur_x3)
        x = self.RM3(x, cur_x2)

        x = self.dropout2d(x)

        x = self.classifier(x)

        out = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=False)
        struct_mask = F.interpolate(struct_mask, (cur_h, cur_w), mode='bilinear', align_corners=False)
        return out, struct_mask
