import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS

@MODELS.register_module()
class StateDualPathHead(BaseDecodeHead):

    def __init__(self, routing='spatial', residual_scale=0.5, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert routing in ['global', 'spatial']
        self.routing = routing
        self.residual_scale = float(residual_scale)
        c = self.channels
        k = self.num_classes
        self.projections = nn.ModuleList([ConvModule(ch, c, 1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU')) for ch in self.in_channels])
        self.semantic_branches = nn.ModuleList([nn.Sequential(ConvModule(2 * c, c, 1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU')), DepthwiseSeparableConvModule(c, c, 3, padding=2, dilation=2, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU'))) for _ in range(3)])
        self.spatial_branches = nn.ModuleList([DepthwiseSeparableConvModule(2 * c, c, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU')) for _ in range(3)])
        self.semantic_classifiers = nn.ModuleList([nn.Conv2d(c, k, 1) for _ in range(3)])
        self.spatial_classifiers = nn.ModuleList([nn.Conv2d(c, k, 1) for _ in range(3)])
        self.local_classifiers = nn.ModuleList([nn.Conv2d(c, k, 1) for _ in range(3)])
        self.routers = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * k + 2, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1)) for _ in range(3)])
        for router in self.routers:
            nn.init.zeros_(router[-1].weight)
            nn.init.zeros_(router[-1].bias)

    def class_context(self, feature, logits):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            probability = F.softmax(logits.float(), dim=1).flatten(2)
            values = feature.float().flatten(2)
            prototypes = values.bmm(probability.transpose(1, 2)) / probability.sum(-1).unsqueeze(1).clamp_min(1e-06)
            context = prototypes.bmm(probability).reshape_as(feature)
        return context

    def route(self, coarse, local, index):
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            p = F.softmax(coarse.float(), dim=1)
            q = F.softmax(local.float(), dim=1)
            uncertainty = 1 - p.amax(1, keepdim=True)
            dx = F.pad((p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean(1, keepdim=True), (0, 1, 0, 0))
            dy = F.pad((p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean(1, keepdim=True), (0, 0, 0, 1))
            state = torch.cat([p, q, (p - q).abs(), uncertainty, dx + dy], dim=1)
            if self.routing == 'global':
                state = F.adaptive_avg_pool2d(state, 1).expand_as(state)
        return F.softmax(self.routers[index](state).float(), dim=1)

    def forward(self, inputs):
        xs = [layer(x) for layer, x in zip(self.projections, self._transform_inputs(inputs))]
        feature = xs[-1]
        logits = self.cls_seg(feature)
        for index in range(2, -1, -1):
            size = xs[index].shape[2:]
            context = resize(self.class_context(feature, logits), size=size, mode='bilinear', align_corners=False)
            coarse_feature = resize(feature, size=size, mode='bilinear', align_corners=False)
            coarse_logits = resize(logits, size=size, mode='bilinear', align_corners=False)
            semantic = self.semantic_branches[index](torch.cat([coarse_feature, context], dim=1))
            spatial = self.spatial_branches[index](torch.cat([coarse_feature, xs[index]], dim=1))
            weights = self.route(coarse_logits, self.local_classifiers[index](xs[index]), index)
            keep, sem, spa = weights.split(1, dim=1)
            semantic_logits = self.semantic_classifiers[index](self.dropout(semantic) if self.dropout is not None else semantic)
            spatial_logits = self.spatial_classifiers[index](self.dropout(spatial) if self.dropout is not None else spatial)
            logits = coarse_logits + self.residual_scale * (sem * semantic_logits + spa * spatial_logits)
            feature = keep * coarse_feature + sem * semantic + spa * spatial
        return logits
