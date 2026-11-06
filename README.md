# GMFSS_Fortuna

The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation, based on https://github.com/98mxr/GMFSS_Fortuna.

This VapourSynth plugin uses PyTorch/CUDA for inference. TensorRT-RTX acceleration is optional.

## Requirements

- Windows x64 and an NVIDIA CUDA-capable GPU with a current NVIDIA driver.
- VapourSynth R55+; use the Python interpreter bundled with your VapourSynth portable distribution.
- 64-bit Python 3.10+ matching the Python version and architecture used by VapourSynth.
- CUDA 12.x-compatible PyTorch and CuPy. This project currently depends on cupy-cuda12x.

The CUDA runtime bundled with a PyTorch wheel does not require a separate full system-wide CUDA Toolkit. The NVIDIA driver must support the selected runtime. CuPy also needs CUDA runtime/compiler components; the steps below install its CUDA 12 component wheels.

## Windows portable installation

The commands below assume the official VapourSynth portable layout, with python.exe and the pip.bat helper in the portable folder alongside vspipe.bat. Open Command Prompt in that folder. **Use python -m pip for every command** so packages are installed into the Python environment VapourSynth actually uses, not another system Python.

### 1. Check Python and VapourSynth

```powershell
python --version
python -c "import vapoursynth as vs; print(vs.__version__)"
```

If VapourSynth does not import, repair the portable VapourSynth installation before continuing.

### 2. Install CUDA-enabled PyTorch

Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) and choose Windows, Pip, Python, and a CUDA version supported by your driver/GPU. For example, CUDA 12.8:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

If the selector recommends a different CUDA index, use that instead. Do not install the CPU-only PyTorch build.

### 3. Install CuPy and CUDA components

This project currently uses CuPy for CUDA 12.x. The ctk extra installs the CUDA components CuPy needs without a separate full system-wide CUDA Toolkit installation:

```powershell
python -m pip install --upgrade "cupy-cuda12x[ctk]"
```

### 4. Install GMFSS_Fortuna

Install this repository:

```powershell
python -m pip install --upgrade "git+https://github.com/ryandash/vs-gmfss_fortuna.git"
```

This installs GMFSS_Fortuna and its remaining Python dependencies, including NumPy. To use the published PyPI package instead:

```powershell
python -m pip install --upgrade vsgmfss-fortuna
```

### 5. Verify the environment

Run from the same portable folder:

```powershell
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
python -c "import cupy; print('CuPy:', cupy.__version__); print('CuPy devices:', cupy.cuda.runtime.getDeviceCount())"
python -c "import vapoursynth; import vsgmfss_fortuna; print('VapourSynth and GMFSS_Fortuna imports OK')"
```

PyTorch must report CUDA available as True and show your NVIDIA GPU. CuPy's device count should be at least 1. Resolve any import or CUDA errors before rendering.

### 6. Use it in a VapourSynth script

The input clip must be RGBH (FP16) or RGBS (FP32):

```python
from vsgmfss_fortuna import gmfss_fortuna

# Replace with your source clip, converted to RGBH or RGBS.
clip = ...
ret = gmfss_fortuna(clip, model=1, scale=1.0)
```

Run the script with the portable distribution's vspipe.bat or a VapourSynth application configured to use that same portable installation.

## Optional: TensorRT-RTX acceleration

The default gmfss_fortuna(clip) path uses PyTorch/CUDA and does **not** require TensorRT. Get the standard path working first.

Torch-TensorRT-RTX, TensorRT-RTX, PyTorch, CUDA, Python, and the NVIDIA driver are version-coupled. Check the current [Torch-TensorRT for RTX installation guide](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html) and its compatibility requirements before installing the optional extra. Installing it may change the PyTorch version selected by pip.

After selecting a compatible combination, install the extra in the portable Python environment:

```powershell
python -m pip install --upgrade "vsgmfss-fortuna[tensorrt-rtx]"
```

If installing from this repository, use:

```powershell
python -m pip install --upgrade "git+https://github.com/ryandash/vs-gmfss_fortuna.git#egg=vsgmfss-fortuna[tensorrt-rtx]"
```

Verify the backend:

```powershell
python -c "import torch_tensorrt, tensorrt_rtx; print('Dynamo compile:', callable(torch_tensorrt.dynamo.compile)); print('Save:', callable(torch_tensorrt.save))"
```

Both checks should print True. If pip proposes replacing a working PyTorch/CUDA install with an incompatible build, cancel and follow the upstream version matrix to install a matching set. Do not blindly upgrade only one of these coupled packages.

Enable TensorRT explicitly:

```python
ret = gmfss_fortuna(
    clip,
    model=1,
    scale=1.0,
    trt=True,
    trt_cache_dir=r"D:\GMFSS_TRT_Cache",
    trt_optimization_level=3,
)
```

The first run exports and compiles the supported components and can take significantly longer than normal inference. Later runs can reuse serialized engines in trt_cache_dir. The cache is specific to the model, input dimensions, precision, GPU, software versions, and builder settings. If TensorRT setup or inference fails, remove trt=True to use the standard PyTorch/CUDA path.

## TensorRT implementation notes

When trt=True, FeatureNet, MetricNet, and GridNet are exported with torch.export and compiled as separate static-shape TensorRT modules through Torch-TensorRT's Dynamo API. GMFlow, IFNet, frame orchestration, and custom CuPy soft-splat kernels remain on the normal PyTorch/CUDA path; TensorRT does not convert the entire model.

trt_max_workspace_size sets the builder workspace per compiled component and defaults to 1 GiB. trt_cache_dir=None keeps compilation in memory. trt_optimization_level controls the builder optimization level (0–5) and defaults to 3.

## Troubleshooting

- **CUDA is not available:** confirm that a CUDA-enabled PyTorch wheel is installed in the portable Python, the NVIDIA driver is installed, and torch.cuda.is_available() returns True.
- **CuPy cannot find CUDA or fails compiling a kernel:** confirm cupy-cuda12x[ctk] is installed and inspect cupy.show_config().
- **VapourSynth imports in one Python but not another:** run all installation and verification commands from the portable folder using its python.exe.
- **TensorRT import or compilation errors:** first verify the standard PyTorch/CUDA path, then check the Torch-TensorRT-RTX compatibility matrix. TensorRT is optional; omit trt=True to bypass it.

References: [VapourSynth Windows portable installation](https://www.vapoursynth.com/doc/installation.html#windows-portable) · [PyTorch](https://pytorch.org/get-started/locally/) · [CuPy](https://docs.cupy.dev/en/stable/install.html) · [Torch-TensorRT for RTX](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html)

See vsgmfss_fortuna/__init__.py for all gmfss_fortuna parameters.
