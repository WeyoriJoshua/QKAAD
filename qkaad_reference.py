#!/usr/bin/env python3
"""Matched quantum-fidelity vs. classical-RBF clean anomaly screening.

Reference implementation for manuscript 7559033. Uses clean validation only
for thresholds. Does not claim adversarial detection performance or clinical use.
"""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 123
REFERENCE_SIZE = 20
GAMMA = 0.25
PERCENTILE = 95.0
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]


def data_splits(seed=SEED):
    ds = load_breast_cancer(as_frame=True)
    x = ds.data[FEATURES].to_numpy(dtype=np.float64)
    y = 1 - ds.target.to_numpy(dtype=int)  # sklearn: malignant=0
    indices = np.arange(len(y))
    tr, remainder = train_test_split(indices, train_size=0.6,
                                      stratify=y, random_state=seed)
    va, te = train_test_split(remainder, train_size=0.5,
                              stratify=y[remainder], random_state=seed)
    assert len(np.unique(np.concatenate([tr, va, te]))) == len(y)
    scaler = MinMaxScaler().fit(x[tr])
    z_train = scaler.transform(x[tr])
    pca = PCA(n_components=4, random_state=seed).fit(z_train)
    out = {
        "train": pca.transform(z_train),
        "validation": pca.transform(scaler.transform(x[va])),
        "test": pca.transform(scaler.transform(x[te])),
        "train_indices": tr, "validation_indices": va, "test_indices": te,
        "train_labels": y[tr], "validation_labels": y[va],
        "test_labels": y[te],
    }
    return out


def classical_kernel(a, b):
    return rbf_kernel(a, b, gamma=GAMMA)


def quantum_kernel_factory():
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(
        feature_dimension=4, reps=1, entanglement="full"))
    return lambda a, b: np.asarray(kernel.evaluate(x_vec=a, y_vec=b))


def anomaly_score(kernel, queries, reference):
    sim = np.asarray(kernel(queries, reference), dtype=np.float64)
    if sim.shape != (len(queries), len(reference)):
        raise ValueError("Kernel shape mismatch")
    if not np.isfinite(sim).all():
        raise ValueError("Non-finite kernel values")
    return 1.0 - sim.mean(axis=1)


def evaluate(name, kernel, parts, reference):
    t0 = time.perf_counter()
    val = anomaly_score(kernel, parts["validation"], reference)
    val_seconds = time.perf_counter() - t0
    threshold = float(np.percentile(val, PERCENTILE))
    t1 = time.perf_counter()
    test = anomaly_score(kernel, parts["test"], reference)
    test_seconds = time.perf_counter() - t1
    vf = val > threshold
    tf = test > threshold
    return {
        "detector": name,
        "reference_size": len(reference),
        "validation_threshold": threshold,
        "validation_fpr": float(vf.mean()),
        "clean_test_fpr": float(tf.mean()),
        "validation_flags": vf.astype(int),
        "test_flags": tf.astype(int),
        "validation_scores": val,
        "test_scores": test,
        "validation_seconds": float(val_seconds),
        "test_seconds": float(test_seconds),
    }


def versions():
    result = {"python": sys.version.split()[0]}
    for package in ["numpy", "pandas", "scikit-learn", "qiskit",
                    "qiskit-machine-learning"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quantum", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parts = data_splits(args.seed)
    # The same reference indices and representation are used by both kernels.
    reference = parts["train"][:REFERENCE_SIZE].copy()
    assert len(reference) == REFERENCE_SIZE

    evaluations = [evaluate("RBF_mean", classical_kernel, parts, reference)]
    if args.quantum:
        evaluations.append(evaluate(
            "Quantum_fidelity_mean", quantum_kernel_factory(), parts, reference))

    scalar_rows = []
    for item in evaluations:
        scalar_rows.append({key: value for key, value in item.items()
                            if not isinstance(value, np.ndarray)})
        pd.DataFrame({
            "validation_score": item["validation_scores"],
            "is_flagged": item["validation_flags"],
        }).to_csv(args.output / (item["detector"] + "_validation.csv"),
                  index=False)
        pd.DataFrame({
            "test_score": item["test_scores"],
            "is_flagged": item["test_flags"],
        }).to_csv(args.output / (item["detector"] + "_clean_test.csv"),
                  index=False)

    pd.DataFrame(scalar_rows).to_csv(
        args.output / "clean_detector_comparison.csv", index=False)
    config = {
        "seed": args.seed, "reference_size": REFERENCE_SIZE,
        "feature_names": FEATURES, "pca_components": 4,
        "rbf_gamma": GAMMA, "calibration_percentile": PERCENTILE,
        "threshold_source": "clean_validation_only",
        "train_size": len(parts["train_indices"]),
        "validation_size": len(parts["validation_indices"]),
        "test_size": len(parts["test_indices"]),
        "label_mapping": {"benign": 0, "malignant": 1},
        "packages": versions(),
        "quantum_executed": bool(args.quantum),
    }
    (args.output / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "split_indices.npz",
        train=parts["train_indices"], validation=parts["validation_indices"],
        test=parts["test_indices"])
    print(pd.DataFrame(scalar_rows).to_string(index=False))
    print("Wrote measured results to:", args.output.resolve())


if __name__ == "__main__":
    main()
