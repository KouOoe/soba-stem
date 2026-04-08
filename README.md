## SOBA: Scattering-matrix-based Optimisation By Automatic-differentiation
SOBA performs quantitative potential reconstruction by 4D-STEM and automatic differentiation scheme implemeted in PyTorch. See our [Microscopy and Microanalysis paper](https://doi.org/10.1093/mam/ozaf111).

## Features
* Quantitative potential optimisation via S-matrix formalism, accounting for dynamical scattering effects
* Optimisation of probe parameters (i.e., aberration coefficients and effective source size for partial coherence)
* OBF-STEM based potential initialisation
* Probe position correction

---

## Repository structure
- `src/soba_stem/`: Core implementation (preprocessing, forward calculation, optimisation, I/O)
- `example/optimisation_run.py`: End-to-end execution script
- `example/optimisation_config*.yaml`: Optimisation configuration files
- `data/`: Sample data and example outputs

## Repository tree

```text
soba-stem/
├── README.md
├── src/
│   └── soba_stem/
│       ├── __init__.py
│       ├── config.py
│       ├── io_utils.py
│       ├── optimisation.py
│       ├── pixelated_data_preprocess.py
│       ├── smatrix_forward_calc.py
│       └── smatrix_preprocess.py
├── example/
│   ├── optimisation_run.py
│   ├── optimisation_config.yaml
│   ├── optimisation_config_test_short.yaml
│   └── optimisation_run.ipynb
└── data/
    ├── experiment/
    └── multislice_frozen_phonon/
```

---
