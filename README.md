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
2. Install CUDA Toolkit 12.9 Update 1 and confirm `nvcc --version`.
3. Create and activate a Python virtual environment that is also used by your VapourSynth installation.
4. Install VapourSynth R55+, then install this package and its base dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

5. For TensorRT-RTX acceleration, install the optional extra:

   ```powershell
   python -m pip install -e ".[tensorrt-rtx]"
   ```

   The extra installs Torch-TensorRT-RTX and its matching TensorRT-RTX Python runtime. Windows support is currently through the Dynamo / `torch.compile` path. See the [Torch-TensorRT for RTX installation guide](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html) and [TensorRT-RTX prerequisites](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/prerequisites.html) for current version and driver requirements.

6. Verify the Python environment:

   ```powershell
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
   python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
   python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
   python -m pip check
   ```

TensorRT-RTX compilation is experimental in Torch-TensorRT. The first inference call may trigger engine compilation. GMFSS includes custom CuPy soft-splat operations and other graph-sensitive operations; TensorRT can only accelerate graph partitions it can capture and convert. Unsupported or graph-broken sections must continue to execute in PyTorch/CUDA. Actual acceleration and numerical parity must be verified on the target GPU. If TensorRT compilation fails, call `gmfss_fortuna(..., trt=False)` to use the regular PyTorch path.

## Installation

For a regular PyTorch installation:

```powershell
python -m pip install -U vsgmfss-fortuna
```

## Usage

```python
from vsgmfss_fortuna import gmfss_fortuna

# PyTorch/CUDA path
ret = gmfss_fortuna(clip)

# Optional TensorRT-RTX path (requires the optional extra above)
ret_trt = gmfss_fortuna(clip, trt=True)
```

See `__init__.py` for the description of the parameters.
