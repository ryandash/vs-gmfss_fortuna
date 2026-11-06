# GMFSS_Fortuna
The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation, based on https://github.com/98mxr/GMFSS_Fortuna.

## Dependencies

- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started)
- [VapourSynth](http://www.vapoursynth.com/) R55+

Inference uses PyTorch with CUDA acceleration.

For Windows installation, see the [VapourSynth releases](https://github.com/HolyWu/vs-gmfss_fortuna/releases) for packaged builds where available. CuPy requires a compatible CUDA runtime.

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
