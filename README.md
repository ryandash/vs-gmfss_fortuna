# GMFSS_Fortuna
The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation, based on https://github.com/98mxr/GMFSS_Fortuna.

## Dependencies

- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started)
- [VapourSynth](http://www.vapoursynth.com/) R55+

The default inference path uses PyTorch with CUDA acceleration. TensorRT-RTX is optional and is not imported unless `trt=True` is requested.

## Installation

Install the regular package as usual:

```powershell
python -m pip install -U vsgmfss-fortuna
```

For the optional TensorRT-RTX backend, install the extra in the same Python environment used by VapourSynth:

```powershell
python -m pip install -U "vsgmfss-fortuna[tensorrt-rtx]"
```

The extra installs [Torch-TensorRT for RTX](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html), which uses the `torch_tensorrt` Python import and the `tensorrt_rtx` runtime. Windows support uses the Dynamo / `torch.compile` path. The package resolver may change the installed PyTorch version to satisfy the selected Torch-TensorRT-RTX wheel; review the proposed changes before accepting them.

**Important:** Torch-TensorRT-RTX wheels are tightly coupled to specific PyTorch, CUDA, TensorRT-RTX, and Python versions. Use the upstream installation matrix and NVIDIA TensorRT-RTX prerequisites to select a matching combination. The current project environment's PyTorch 2.7.1 + CUDA 12.8 combination should not be assumed compatible with the latest Torch-TensorRT-RTX release. The CUDA Toolkit and driver requirements are separate from the CUDA runtime bundled with PyTorch.

Verify the environment after installation:

```powershell
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
python -c "import torch_tensorrt, tensorrt_rtx; print('Torch-TensorRT:', torch_tensorrt.__version__); print('TensorRT-RTX:', tensorrt_rtx.__version__)"
```

## TensorRT-RTX implementation

When `trt=True`, the plugin compiles FeatureNet, MetricNet, and GridNet as separate static-shape TensorRT modules through `torch.compile(..., backend="tensorrt")`. The first inference triggers compilation. Intermediate tensors remain on the GPU.

Feature extraction for the two input frames is batched into one FeatureNet call. GMFlow is also invoked once in its native bidirectional mode, sharing the image-backbone pass and producing forward/backward flow together. These changes reduce duplicated module invocations; benchmark on the target GPU because the larger batch can change peak memory and kernel selection. GMFlow, IFNet, GMFSS frame orchestration, and custom CuPy soft-splat kernels remain on the normal PyTorch/CUDA path. This conservative component allowlist avoids capturing the more complex optical-flow and custom-kernel portions of the model. TensorRT may still leave unsupported operators in PyTorch, depending on the installed backend's lowering and fallback behavior.

The `trt_max_workspace_size` argument sets the per-component tactic workspace when supported by the installed Torch-TensorRT version. It defaults to 1 GiB. `trt_cache_dir` optionally selects a directory for the cache file; the plugin creates the directory and appends `gmfss_timing.cache`. For example, `gmfss_fortuna(clip, trt=True, trt_cache_dir=user_directory + "/ConvertedVideos")`. `trt_optimization_level` controls TensorRT's builder search effort (0–5) and defaults to 3, the upstream default. Higher levels can improve the resulting engine but increase initial build time; level 0 prioritizes faster builds. The cache path is passed as Torch-TensorRT's `timing_cache_path` when that setting is exposed. Note that current upstream Torch-TensorRT documentation says timing caches are not used by TensorRT-RTX; this option may therefore have no effect with the RTX backend. It is not a serialized-engine cache. Torch-TensorRT options are version-dependent; the implementation only passes settings exposed by the installed version.

TensorRT-RTX acceleration is opt-in. If engine compilation or inference fails, remove `trt=True` to use the standard PyTorch/CUDA path. A successful render does not by itself prove that all operations were converted to TensorRT; verify engine compilation and benchmark against the PyTorch baseline.


### Troubleshooting a missing TensorRT backend

After installing the RTX extra, verify that importing Torch-TensorRT succeeds and registers the `tensorrt` Dynamo backend:

```powershell
python -c "import torch; import torch_tensorrt; import tensorrt_rtx; print('Torch:', torch.__version__); print('Torch-TensorRT:', torch_tensorrt.__file__); print('TensorRT-RTX:', tensorrt_rtx.__file__); print('Backends:', torch._dynamo.list_backends())"
```

The backend list should include `tensorrt`. If `torch_tensorrt` imports as a namespace package (for example, `torch_tensorrt.__file__` is `None`) or its package files are missing, the wheel may have been incompletely installed. Reinstall the **same version already selected for the environment** without changing its dependencies:

```powershell
python -m pip install --pre torch torchvision torch_tensorrt_rtx --extra-index-url https://download.pytorch.org/whl/nightly/cu132
```

## Usage

```python
from vsgmfss_fortuna import gmfss_fortuna

# Standard PyTorch/CUDA inference
ret = gmfss_fortuna(clip)

# Optional TensorRT-RTX acceleration
ret_trt = gmfss_fortuna(clip, trt=True)
```

See `vsgmfss_fortuna/__init__.py` for all available parameters.
