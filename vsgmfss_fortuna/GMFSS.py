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
        self.model_dir = model_dir
        self.scale = scale
        self._tensorrt_compiled = False

    def compile_tensorrt(
        self,
        input_height,
        input_width,
        cache_dir=None,
        workspace_size=1 << 30,
        optimization_level=3,
    ):
        """Compile selected modules and persist/reload their TensorRT engines.

        Cache entries are static-shape and keyed by model, dtype, GPU,
        software versions, and builder settings. GMFlow, IFNet, frame
        orchestration, and CuPy soft-splat remain eager.
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

        import hashlib
        import json
        import tempfile
        import warnings

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        gpu_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        trt_version = getattr(torch_tensorrt, "__version__", "unknown")
        rtx_version = getattr(tensorrt_rtx, "__version__", "unknown")

        # FeatureNet consumes the two frames as one batch. MetricNet operates
        # at half input resolution; GridNet receives its four pyramid inputs
        # at 1x, 1/2x, and 1/4x of that half-resolution image.
        h2, w2 = input_height // 2, input_width // 2
        h4, w4 = h2 // 2, w2 // 2
        h8, w8 = h4 // 2, w4 // 2
        examples = {
            "feat_ext": (
                torch.randn((2, 3, input_height, input_width), device=device, dtype=dtype).contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            "metricnet": (
                torch.randn((1, 3, h2, w2), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
                torch.randn((1, 3, h2, w2), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
                torch.randn((1, 2, h2, w2), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
                torch.randn((1, 2, h2, w2), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
            ),
            "fusionnet": (
                torch.randn((1, 12 if self.model_type == "base" else 9, h2, w2), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
                torch.randn((1, 128, h2, w2), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
                torch.randn((1, 256, h4, w4), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
                torch.randn((1, 384, h8, w8), device=device, dtype=dtype).contiguous(memory_format=torch.channels_last),
            ),
        }

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        for name in ("feat_ext", "metricnet", "fusionnet"):
            component = getattr(self, name).eval()
            component_inputs = examples[name]
            checkpoint_names = {
                "feat_ext": f"feat_{self.model_type}.pkl",
                "metricnet": f"metric_{self.model_type}.pkl",
                "fusionnet": f"fusionnet_{self.model_type}.pkl",
            }
            checkpoint_path = os.path.join(self.model_dir, checkpoint_names[name])
            checkpoint_hash = hashlib.sha256()
            with open(checkpoint_path, "rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    checkpoint_hash.update(chunk)

            cache_identity = {
                "component": name,
                "weights_sha256": checkpoint_hash.hexdigest(),
                "model": self.model_type,
                "shape": [list(t.shape) for t in component_inputs],
                "dtype": str(dtype),
                "gpu": gpu_name,
                "compute_capability": capability,
                "torch": torch.__version__,
                "torch_tensorrt": trt_version,
                "tensorrt_rtx": rtx_version,
                "workspace_size": int(workspace_size),
                "optimization_level": int(optimization_level),
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_identity, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            engine_path = (
                os.path.join(cache_dir, f"gmfss_{name}_{cache_key}.ep")
                if cache_dir is not None else None
            )

            if engine_path is not None and os.path.isfile(engine_path):
                try:
                    compiled_component = torch.export.load(engine_path).module().eval()
                    setattr(self, name, compiled_component)
                    continue
                except Exception as exc:
                    warnings.warn(
                        f"Could not load cached TensorRT engine {engine_path}: {exc}. "
                        "Rebuilding it.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    try:
                        os.remove(engine_path)
                    except OSError:
                        pass

            if name == "metricnet":
                # MetricNet's backwarp uses a module-global shape cache.
                # Populate it before export so tracing does not capture a
                # dictionary mutation or a cache-miss branch.
                prepare_backwarp_grid(component_inputs[2])

            exported = torch.export.export(component, component_inputs)
            compile_options = {
                "workspace_size": int(workspace_size),
                "optimization_level": int(optimization_level),
                "num_avg_timing_iters": 1,
                "enabled_precisions": {dtype},
            }
            compiled_component = torch_tensorrt.dynamo.compile(
                exported,
                component_inputs,
                device=device,
                min_block_size=1,
                use_explicit_typing=True,
                **compile_options,
            ).eval()

            if engine_path is not None:
                # Save to a temporary file first so an interrupted write never
                # leaves a partial cache entry that looks valid on next launch.
                fd, temporary_path = tempfile.mkstemp(
                    prefix=".gmfss_engine_", suffix=".ep", dir=cache_dir
                )
                os.close(fd)
                try:
                    torch_tensorrt.save(
                        compiled_component,
                        temporary_path,
                        output_format="exported_program",
                        inputs=component_inputs,
                    )
                    os.replace(temporary_path, engine_path)
                finally:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)

                # Use the serialized artifact for this run too, ensuring that
                # cache load and cache creation follow the same execution path.
                compiled_component = torch.jit.load(engine_path, map_location=device).eval()

            setattr(self, name, compiled_component)

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
