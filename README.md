# GMFSS_Fortuna
The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation, based on https://github.com/98mxr/GMFSS_Fortuna.

## Dependencies

- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 2.0 or newer
- [VapourSynth](http://www.vapoursynth.com/) R55+

The normal PyTorch/CUDA inference path does not require TensorRT. TensorRT acceleration is optional.

## Windows 11 / RTX 5000-series (Blackwell)

The RTX 5070 Ti uses NVIDIA Blackwell (compute capability 12.0). For the optional TensorRT-RTX backend, use a supported NVIDIA driver and CUDA Toolkit. NVIDIA currently lists CUDA Toolkit 12.9 Update 1 or CUDA 13.4 as TensorRT-RTX prerequisites. This project currently uses the CuPy CUDA 12.x package, so use the CUDA 12.9 Update 1 TensorRT-RTX package unless you deliberately migrate the CuPy dependency to a CUDA 13.x build.

1. Install the current NVIDIA driver.
2. Install CUDA Toolkit 12.9 Update 1 and confirm nvcc --version.
3. Create and activate a Python virtual environment that is also used by your VapourSynth installation.
4. Install VapourSynth R55+, then install this package and its base dependencies:

       python -m pip install --upgrade pip
       python -m pip install -e .

5. For TensorRT-RTX acceleration, install the optional extra:

       python -m pip install -e ".[tensorrt-rtx]"

   The extra installs Torch-TensorRT-RTX and its matching TensorRT-RTX Python runtime. Windows support is currently through the Dynamo / torch.compile path. See the Torch-TensorRT for RTX installation guide and TensorRT-RTX prerequisites for current version and driver requirements.

6. Verify the Python environment:

       python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
       python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
       python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
       python -m pip check

## TensorRT component compilation

When trt=True, the plugin compiles FeatureNet, MetricNet, and GridNet as separate static-shape TensorRT modules. GMFlow (including correlation, warping, and convex flow upsampling), IFNet, GMFSS frame orchestration, and custom CuPy soft-splat operations remain in eager PyTorch. Intermediate tensors stay on the GPU; component boundaries do not require host copies.

This is an intentionally conservative allowlist. GMFlow contains large correlation and flow-refinement graphs that have triggered expensive or unsuccessful TensorRT tactic builds on the target RTX GPU. IFNet remains eager until it has independent correctness and performance coverage. Only components selected by the allowlist are wrapped with torch.compile.

The first inference triggers compilation of the selected component engines. The plugin explicitly sets each engine's permitted precision to match that component's parameter dtype (FP16 for RGBH input and FP32 for RGBS input), rather than relying on Torch-TensorRT's default precision setting.

The trt_max_workspace_size argument controls the maximum TensorRT tactic workspace in bytes for each compiled component (default: 1 GiB). A smaller workspace may restrict tactic choices and reduce runtime performance. Unsupported operators inside a selected component may still execute through PyTorch fallback, so a successful frame render alone does not prove every operation was converted to TensorRT.

TensorRT-RTX is distinct from the standard TensorRT Python package. The optional extra installs Torch-TensorRT-RTX; the Python import name remains torch_tensorrt for either distribution. Check the installed package flavor and its PyTorch/CUDA compatibility before using the RTX extra. Actual acceleration, numerical parity, memory use, and performance must be verified on the target GPU. If TensorRT compilation fails, use gmfss_fortuna(clip, trt=False) for the regular PyTorch path.

## Installation

For a regular PyTorch installation:

    python -m pip install -U vsgmfss-fortuna

## Usage

    from vsgmfss_fortuna import gmfss_fortuna

    # PyTorch/CUDA path
    ret = gmfss_fortuna(clip)

    # Optional TensorRT-RTX component path (requires the optional extra above)
    ret_trt = gmfss_fortuna(clip, trt=True)

See __init__.py for the description of the parameters.
