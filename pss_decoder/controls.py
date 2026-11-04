import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from .decoder import StateDualPathHead


@MODELS.register_module()
class ProgressiveResidualHead(BaseDecodeHead):

    def __init__(self, residual_scale=0.5, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        c = self.channels
        self.residual_scale = float(residual_scale)
        self.projections = nn.ModuleList([ConvModule(ch, c, 1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU')) for ch in self.in_channels])
        self.refiners = nn.ModuleList([DepthwiseSeparableConvModule(2 * c, c, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU')) for _ in range(3)])
        self.residual_classifiers = nn.ModuleList([nn.Conv2d(c, self.num_classes, 1) for _ in range(3)])

    def forward(self, inputs):
        xs = [layer(x) for layer, x in zip(self.projections, self._transform_inputs(inputs))]
        semantic = xs[-1]
        logits = self.cls_seg(semantic)
        for i in range(2, -1, -1):
            semantic = self.refiners[i](torch.cat([xs[i], resize(semantic, size=xs[i].shape[2:], mode='bilinear', align_corners=False)], dim=1))
            residual = self.residual_classifiers[i](self.dropout(semantic) if self.dropout is not None else semantic)
            logits = resize(logits, size=xs[i].shape[2:], mode='bilinear', align_corners=False) + self.residual_scale * residual
        return logits

@MODELS.register_module()
class UniformRoutingHead(StateDualPathHead):

    def route(self, coarse, local, index):
        return torch.full((coarse.shape[0], 3, *coarse.shape[2:]), 1 / 3, device=coarse.device, dtype=torch.float32)

class MaskedCorrectionHead(StateDualPathHead):
    disabled_index = None

    def route(self, coarse, local, index):
        weights = super().route(coarse, local, index)
        mask = weights.new_ones((1, 3, 1, 1))
        mask[:, self.disabled_index] = 0
        weights = weights * mask
        total = weights.sum(1, keepdim=True)
        return torch.where(total > 1e-12, weights / total.clamp_min(1e-12), mask.expand_as(weights) / 2)

@MODELS.register_module()
class NoSemanticCorrectionHead(MaskedCorrectionHead):
    disabled_index = 1

@MODELS.register_module()
class NoSpatialCorrectionHead(MaskedCorrectionHead):
    disabled_index = 2

@MODELS.register_module()
class InitialGuidanceControlHead(StateDualPathHead):

    def class_context(self, feature, logits):
        guidance = resize(self._initial_guidance, size=feature.shape[2:], mode='bilinear', align_corners=False)
        return super().class_context(feature, guidance)

    def route(self, coarse, local, index):
        guidance = resize(self._initial_guidance, size=coarse.shape[2:], mode='bilinear', align_corners=False)
        return super().route(guidance, local, index)

    def forward(self, inputs):

        def capture(module, args, output):
            self._initial_guidance = output
        self._initial_guidance = None
        hook = self.conv_seg.register_forward_hook(capture)
        try:
            return super().forward(inputs)
        finally:
            hook.remove()
            self._initial_guidance = None

@MODELS.register_module()
class ConfidenceRoutingHead(StateDualPathHead):

    def route(self, coarse, local, index):
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            probability = F.softmax(coarse.float(), dim=1)
            batch, classes, height, width = probability.shape
            uncertainty = 1 - probability.amax(1, keepdim=True)
            before = probability.new_zeros((batch, 3 * classes, height, width))
            state = torch.cat([before, uncertainty, torch.zeros_like(uncertainty)], dim=1)
            if self.routing == 'global':
                state = F.adaptive_avg_pool2d(state, 1).expand_as(state)
        return F.softmax(self.routers[index](state).float(), dim=1)

@MODELS.register_module()
class StageConstantRoutingHead(StateDualPathHead):

    def route(self, coarse, local, index):
        weights = self.routers[index][-1].bias.float().softmax(0)
        return weights.view(1, 3, 1, 1).expand(coarse.shape[0], 3, *coarse.shape[2:])

@MODELS.register_module()
class InitialContextConstantHead(StageConstantRoutingHead):

    def class_context(self, feature, logits):
        assert self._initial_context_logits is not None
        guidance = resize(self._initial_context_logits, size=feature.shape[2:], mode='bilinear', align_corners=False)
        return super().class_context(feature, guidance)

    def forward(self, inputs):
        self._initial_context_logits = None

        def capture(module, args, output):
            assert self._initial_context_logits is None
            self._initial_context_logits = output
        hook = self.conv_seg.register_forward_hook(capture)
        try:
            return super().forward(inputs)
        finally:
            hook.remove()
            self._initial_context_logits = None

def acf_context(feature, logits):
    with torch.autocast(device_type=feature.device.type, enabled=False):
        probability = F.softmax(logits.float(), dim=1).flatten(2)
        values = feature.float().flatten(2)
        centers = values.bmm(probability.transpose(1, 2))
        centers = centers / probability.sum(-1).unsqueeze(1).clamp_min(1e-06)
        return centers.bmm(probability).reshape_as(feature)

@MODELS.register_module()
class ACFControlledAdapterHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        c = self.channels

        def project(cin):
            return ConvModule(cin, c, 1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU'))
        self.projections = nn.ModuleList([project(ch) for ch in self.in_channels])
        self.fusion = project(len(self.in_channels) * c)
        self.coarse_classifier = nn.Conv2d(c, self.num_classes, 1)
        self.context_projection = project(c)
        self.refine = project(2 * c)

    def forward(self, inputs):
        xs = [layer(x) for layer, x in zip(self.projections, self._transform_inputs(inputs))]
        size = xs[0].shape[2:]
        feature = self.fusion(torch.cat([resize(x, size=size, mode='bilinear', align_corners=False) for x in xs], dim=1))
        coarse = self.coarse_classifier(feature)
        context = self.context_projection(acf_context(feature, coarse))
        return self.cls_seg(self.refine(torch.cat([feature, context], dim=1)))

@MODELS.register_module()
class InitialContextDynamicHead(InitialContextConstantHead):
    route = StateDualPathHead.route

@MODELS.register_module()
class SameSourceConstantHead(StageConstantRoutingHead):

    def __init__(self, source='context', **kwargs):
        assert source in ('context', 'shallow', 'mixed')
        self.source = source
        super().__init__(**kwargs)

    def forward(self, inputs):
        xs = [layer(x) for layer, x in zip(self.projections, self._transform_inputs(inputs))]
        feature = xs[-1]
        logits = self.cls_seg(feature)
        for index in range(2, -1, -1):
            size = xs[index].shape[2:]
            context = resize(self.class_context(feature, logits), size=size, mode='bilinear', align_corners=False)
            coarse_feature = resize(feature, size=size, mode='bilinear', align_corners=False)
            coarse_logits = resize(logits, size=size, mode='bilinear', align_corners=False)
            semantic_source = xs[index] if self.source == 'shallow' else context
            spatial_source = context if self.source == 'context' else xs[index]
            semantic = self.semantic_branches[index](torch.cat([coarse_feature, semantic_source], dim=1))
            spatial = self.spatial_branches[index](torch.cat([coarse_feature, spatial_source], dim=1))
            weights = self.route(coarse_logits, self.local_classifiers[index](xs[index]), index)
            keep, sem, spa = weights.split(1, dim=1)
            semantic_logits = self.semantic_classifiers[index](self.dropout(semantic) if self.dropout is not None else semantic)
            spatial_logits = self.spatial_classifiers[index](self.dropout(spatial) if self.dropout is not None else spatial)
            logits = coarse_logits + self.residual_scale * (sem * semantic_logits + spa * spatial_logits)
            feature = keep * coarse_feature + sem * semantic + spa * spatial
        return logits

@MODELS.register_module()
class SimpleFeatureFusionHead(StateDualPathHead):

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
            feature = keep * coarse_feature + sem * semantic + spa * spatial
            logits = self.cls_seg(feature)
        return logits
