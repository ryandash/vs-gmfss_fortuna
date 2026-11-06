# GMFSS_Fortuna
The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation, based on https://github.com/98mxr/GMFSS_Fortuna.

## Dependencies

- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started)
- [VapourSynth](http://www.vapoursynth.com/) R55+

Inference uses PyTorch with CUDA acceleration.

For Windows installation, see the [VapourSynth releases](https://github.com/HolyWu/vs-gmfss_fortuna/releases) for packaged builds where available. CuPy 14 requires CUDA 12 or later and NumPy 2 or later. The package dependency installs the CUDA 12.x CuPy build; use the matching CuPy package for other supported CUDA major versions.

## Installation

```bash
python -m pip install -U vsgmfss-fortuna
```

## Usage

```python
from vsgmfss_fortuna import gmfss_fortuna

ret = gmfss_fortuna(clip)
```

See `vsgmfss_fortuna/__init__.py` for the available parameters.
