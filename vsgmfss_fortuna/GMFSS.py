import os
from dataclasses import fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FeatureNet import FeatureNet
from .gmflow.gmflow import GMFlow
from .IFNet_HDv3 import IFNet
from .MetricNet import MetricNet, prepare_backwarp_grid
from .softsplat import softsplat as warp

torch.fx.wrap('warp')


class GMFSS(nn.Module):
    def __init__(self, model_dir, model_type, scale, ensemble):
        super(GMFSS, self).__init__()
        if model_type == 'base':
            from .FusionNet_b import GridNet
        else:
            from .FusionNet_u import GridNet
            self.ifnet = IFNet(ensemble)
            self.ifnet.load_state_dict(torch.load(os.path.join(model_dir, 'rife.pkl'), map_location='cpu'))
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()
        self.flownet.load_state_dict(torch.load(os.path.join(model_dir, 'flownet.pkl'), map_location='cpu'))
        self.metricnet.load_state_dict(torch.load(os.path.join(model_dir, f'metric_{model_type}.pkl'), map_location='cpu'))
        self.feat_ext.load_state_dict(torch.load(os.path.join(model_dir, f'feat_{model_type}.pkl'), map_location='cpu'))
        self.fusionnet.load_state_dict(torch.load(os.path.join(model_dir, f'fusionnet_{model_type}.pkl'), map_location='cpu'))
        self.model_type = model_type
        self.scale = scale
        self._tensorrt_compiled = False

    def compile_tensorrt(self, workspace_size=1 << 30, timing_cache_path=None):
        """Enable the optional Torch-TensorRT-RTX backend for selected modules.

        GMFlow, IFNet, frame orchestration, and CuPy soft-splat remain eager.
        Only FeatureNet, MetricNet, and GridNet are compiled. This deliberately
        limits graph capture to convolutional modules with tensor-only inputs
        and outputs; unsupported operators in those graphs may still fall back
        to PyTorch according to Torch-TensorRT's Dynamo settings.
        """
        if self._tensorrt_compiled:
            return

        try:
            import tensorrt_rtx  # noqa: F401
            import torch_tensorrt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT-RTX was requested, but its Python packages are missing. "
                "Install the 'tensorrt-rtx' optional dependency."
            ) from exc

        # Torch-TensorRT's Dynamo settings evolve between releases. Only pass
        # options exposed by the installed version to avoid version-specific
        # keyword failures.
        settings_type = getattr(
            getattr(torch_tensorrt, "dynamo", None),
            "CompilationSettings",
            None,
        )
        try:
            available_settings = {field.name for field in fields(settings_type)}
        except (TypeError, AttributeError):
            available_settings = set()

        requested_options = {
            "workspace_size": int(workspace_size),
            "optimization_level": 1,
            "num_avg_timing_iters": 1,
        }
        if timing_cache_path is not None:
            requested_options["timing_cache_path"] = str(timing_cache_path)

        base_options = {
            name: value
            for name, value in requested_options.items()
            if name in available_settings
        }

        # Do not compile GMFlow or IFNet here: their correlation / flow
        # refinement graphs are substantially more complex and remain on the
        # tested PyTorch CUDA path.
        for name in ("feat_ext", "metricnet", "fusionnet"):
            component = getattr(self, name)
            component_options = dict(base_options)

            if "enabled_precisions" in available_settings:
                try:
                    component_dtype = next(component.parameters()).dtype
                except StopIteration:
                    component_dtype = torch.float32
                component_options["enabled_precisions"] = {component_dtype}

            setattr(
                self,
                name,
                torch.compile(
                    component,
                    backend="tensorrt",
                    dynamic=False,
                    options=component_options,
                ),
            )

        self._tensorrt_compiled = True

    def reuse(self, img0, img1):
        # FeatureNet is fully convolutional and both frames have identical
        # shapes. A single batch-2 invocation reduces Python/dispatcher and
        # kernel-launch overhead compared with two batch-1 calls.
        features = self.feat_ext(torch.cat((img0, img1), dim=0))
        feat_ext0, feat_ext1 = [], []
        for feature in features:
            feat0, feat1 = torch.chunk(feature, 2, dim=0)
            feat_ext0.append(feat0)
            feat_ext1.append(feat1)

        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear")
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear")

        if self.scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor = self.scale, mode="bilinear")
            imgf1 = F.interpolate(img1, scale_factor = self.scale, mode="bilinear")
        else:
            imgf0 = img0
            imgf1 = img1

        # GMFlow has a native bidirectional mode. It shares the image
        # backbone pass and derives the reverse global correlation by
        # transposing the forward correlation matrix.
        flow_bidir = self.flownet(imgf0, imgf1, pred_bidir_flow=True)
        flow01, flow10 = torch.chunk(flow_bidir, 2, dim=0)

        if self.scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor = 1. / self.scale, mode="bilinear") / self.scale
            flow10 = F.interpolate(flow10, scale_factor = 1. / self.scale, mode="bilinear") / self.scale

        # Populate the grid cache before Dynamo traces the compiled MetricNet.
        prepare_backwarp_grid(flow01)
        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def forward(self, img0, img1, timestep):
        reuse_things = self.reuse(img0, img1)
        flow01, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][1], reuse_things[4][2]
        flow10, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][1], reuse_things[5][2]

        F1t = timestep * flow01
        F2t = (1-timestep) * flow10

        Z1t = timestep * metric0
        Z2t = (1-timestep) * metric1

        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear")
        I1t = warp(img0, F1t, Z1t, strMode='soft')
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear")
        I2t = warp(img1, F2t, Z2t, strMode='soft')

        if self.model_type == 'union':
            rife = self.ifnet(img0, img1, timestep)

        feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

        F1td = F.interpolate(F1t, scale_factor = 0.5, mode="bilinear") * 0.5
        Z1d = F.interpolate(Z1t, scale_factor = 0.5, mode="bilinear")
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        F2td = F.interpolate(F2t, scale_factor = 0.5, mode="bilinear") * 0.5
        Z2d = F.interpolate(Z2t, scale_factor = 0.5, mode="bilinear")
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

        F1tdd = F.interpolate(F1t, scale_factor = 0.25, mode="bilinear") * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor = 0.25, mode="bilinear")
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        F2tdd = F.interpolate(F2t, scale_factor = 0.25, mode="bilinear") * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor = 0.25, mode="bilinear")
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

        out = self.fusionnet(torch.cat([img0, I1t, I2t, img1] if self.model_type == 'base' else [I1t, rife, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        return torch.clamp(out, 0, 1)
