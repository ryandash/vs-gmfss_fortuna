# GMFSS_Fortuna

The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation, based on [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna).

## Dependencies

- [CuPy (CUDA 12.x)](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [VapourSynth](https://github.com/vapoursynth/vapoursynth/releases)

The default inference path uses PyTorch with CUDA acceleration. TensorRT-RTX is optional and is only required when `trt=True` is requested.

## Installation

Run the following commands using the Python interpreter bundled with your VapourSynth portable installation.

First, install a CUDA-enabled version of PyTorch. Select the appropriate CUDA build for your system using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

Install CuPy for CUDA 12.x:

```powershell
python -m pip install -U "cupy-cuda12x[ctk]"
```

Standard installation:

```powershell
python -m pip install -U "vsgmfss-fortuna @ git+https://github.com/ryandash/vs-gmfss_fortuna.git"
```

With TensorRT-RTX:

```powershell
python -m pip install -U "vsgmfss-fortuna[tensorrt-rtx] @ git+https://github.com/ryandash/vs-gmfss_fortuna.git"
```

**Important:** Torch-TensorRT-RTX wheels are tightly coupled to specific PyTorch, CUDA, TensorRT-RTX, and Python versions. Do not assume that the latest release is compatible with your installed PyTorch or CUDA environment. Consult the [Torch-TensorRT-RTX documentation](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html) and verify compatibility before installing.

During development, the following command was used to install the nightly versions of PyTorch, TorchVision, and Torch-TensorRT-RTX:

```powershell
python -m pip install --pre torch torchvision torch_tensorrt_rtx --extra-index-url https://download.pytorch.org/whl/nightly/cu132
```

This is a development example, not a recommended universal installation command. The package versions and CUDA build must match your environment.

## Usage

```python
from vsgmfss_fortuna import gmfss_fortuna

clip = gmfss_fortuna(
    clip,
    trt=False,
    model=1,
    scale=1.0,
    num_streams=1,
    trt_optimization_level=3,
)
```

Set `trt=True` if using TensorRT-RTX version

See `vsgmfss_fortuna/__init__.py` for all available parameters.
