# QKAAD: Quantum Kernel-Based Adversarial Anomaly Detection

Research code accompanying manuscript **7559033**, *Diagnosis Using Variational Quantum Classifiers and Quantum Kernel-Based Adversarial Anomaly Detection* (Quantum Engineering, major revision).

## Status
This directory is a **code release in progress**, not evidence of clinical readiness or quantum advantage. The recorded experiments do **not** establish consistent superiority of quantum fidelity kernels over matched classical RBF kernels. Raw experimental notebooks, trained model objects, and large result archives are maintained separately until they have been audited and prepared for release.

## Included
- `qkaad_reference.py`: dependency-light, reproducible matched detector protocol, train-only preprocessing, validation-only thresholds, and optional Qiskit fidelity kernel.
- `requirements.txt`: core dependencies and optional quantum packages.
- `README.md`: setup, experimental caveats, and manuscript context.

## Run
```bash
python -m pip install -r requirements.txt
python qkaad_reference.py --output results --quantum
```
For a classical-only smoke test, omit `--quantum`. Qiskit Machine Learning and Qiskit Algorithms are needed for the quantum option.

The dataset is loaded from `sklearn.datasets.load_breast_cancer` (569 observations; 30 feature columns). We retain ten **mean** diagnostic features, with benign=0 and malignant=1. We use a stratified 60/20/20 split, train-fitted MinMax scaling, and train-fitted PCA to four components. Both detectors use the same training reference examples, held-out clean validation for a 95th-percentile threshold, and the same clean test set.

The classical RBF bandwidth is `gamma=0.25`; the quantum kernel is `FidelityQuantumKernel` with four-qubit `ZZFeatureMap(reps=1)`. Both compute `1-mean(kernel(query, reference))`. No test examples are used to select parameters or calibrate thresholds.

## Interpretation and limitations
- The script reports **clean false-positive behavior only**, not adversarial precision/recall or attack-detection superiority.
- Full classifier-directed attacks, eight-variant ablations, repeated-seed VQC training, finite-shot inference checks and scaling experiments are documented in the research notebook; their original exported results must be audited prior to public release.
- Scores obtained with Qiskit simulators are **not measurements on quantum hardware**. Kernel evaluation may be far more computationally expensive than a classical RBF kernel on small tabular data.
- PCA-coordinate perturbations are not validated as physiologically realistic medical measurements. The work does not provide clinical validation.

## Dataset
Wolberg, Street & Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

## Attribution
Standard algorithms and library implementations are credited to their respective original authors. Historical notebook material that predates this study is not claimed as original algorithmic invention.

## Reproducibility
Generated CSV/JSON outputs include the experiment settings and package versions. No performance results are manually hard-coded into the reference implementation. Keep machine-specific artifacts and data with personal information out of public repositories.
