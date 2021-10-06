<h1 align="center">
<p>SWAM :part_alternation_mark:</p>
<p align="center">
<img alt="GitHub" src="https://img.shields.io/github/license/cross-caps/AFLI?color=green&logo=GNU&logoColor=green">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="pytorch" src="https://img.shields.io/badge/pytorch-%3D2.5.0-orange?logo=pytorch">
<img alt="Plotly" src="https://img.shields.io/badge/Plotly-%3D5.3.1-green?logo=plotly">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>

<h2 align="center">
<p>Spectral visualization for raw Waveform based deep Acoustic Models</p>
</h2>

Code to reproduce plots from the paper:
Time-Frequency and Geometric Analysis of Task-dependent learning in Raw Waveform based Acoustic Models

<h4 align="centre"> 
    <p align="centre" > Response Spectra for Speech Classification</p>
    <img src="https://github.com/Cross-Caps/SWAM/blob/main/Plots/speech_classification_3dplots.gif" width="2000" height="200" />
    <p align="centre" > Response Spectra for Male Speaker Classification</p>
    <img src="https://github.com/Cross-Caps/SWAM/blob/main/Plots/speaker_male_classification_3dplots.gif" width="2000" height="200" />
    <p align="centre" > Response Spectra for Female Speaker Classification</p>
    <img src="https://github.com/Cross-Caps/SWAM/blob/main/Plots/speaker_female_classification_3dplots.gif" width="2000" height="200" />
</h4>

## Visualizations

- For implementation of short-time response spectra (STRS) and cumulative response spectra (CRS) maps; see [Script](./acousvis/spectral_properties.py)
- For implementation of various geometric properties; see [Script](./acousvis/geometric_properties.py)
- A demo on how to plot STRS and CRS maps is available in [Notebook](./acousvis/Visualization_demo.ipynb)

## References & Credits

1. [Dictionary properties](https://www.ux.uis.no/~karlsk/dle/dictprop.m) K. Skretting and K. Engan, “Learned dictionaries for sparse image representation: properties and results,” in Wavelets and Sparsity XIV. International Society for Optics and Photonics, 2011, vol. 8138, pp. 404 – 417, SPIE
2. [WV Contours]() Samer A. Abdallah and Mark D. Plumbley,   “If the independent components of natural images are edges, what are the independent components of natural sounds?,” in International Workshop on Independent Component Analysis and Blind Separation (ICA), September 2001, pp. 534–539.

## Contact 

Devansh Gupta <devansh19160@iiitd.ac.in>

Vinayak Abrol <abrol@iiitd.ac.in>
