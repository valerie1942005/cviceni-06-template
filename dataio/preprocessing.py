from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def standardize_fit(x_train: np.ndarray) -> dict:
    """Spocita parametry z-skorove standardizace **jen z trenovacich dat**."""
    x_train = np.asarray(x_train, dtype=np.float64)
    assert x_train.ndim == 2, "x_train musí být 2D matice."

    # Průměr a populační směrodatná odchylka (ddof=0) po sloupcích
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=0)

    return {"mean": mean, "std": std}


def standardize_apply(x: np.ndarray, params: dict) -> np.ndarray:
    """Aplikuje ulozene parametry standardizace (z-skore) na matici X."""
    assert "mean" in params and "std" in params, "params musí obsahovat klíče 'mean' a 'std'."

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(params["mean"], dtype=np.float64)
    std = np.asarray(params["std"], dtype=np.float64)

    assert x.shape[1] == len(mean), "Počet příznaků v matici x neodpovídá uloženým parametrům."

    # Nahrazení nulové směrodatné odchylky jedničkou pro bezpečnou divizi
    std_safe = np.where(std == 0.0, 1.0, std)

    return (x - mean) / std_safe


def subsample_imbalance(
        x: np.ndarray,
        y: np.ndarray,
        minority_ratio: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Podvzorkovanim vytvori silne nevyvazeny dataset (~90/10)."""
    assert 0 < minority_ratio < 0.5, "minority_ratio musí být striktně mezi 0 a 0.5."
    assert len(np.unique(y)) >= 2, "Pole y musí obsahovat vzorky obou tříd."

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    counts = np.bincount(y)

    # OPRAVA: Zajištění správné identifikace tříd i při remíze u vyvážených dat
    min_cls = np.argmin(counts)
    maj_cls = 1 - min_cls

    idx_min = np.flatnonzero(y == min_cls)
    idx_maj = np.flatnonzero(y == maj_cls)

    n_min = len(idx_min)
    n_maj = len(idx_maj)

    r = minority_ratio
    current_ratio = n_min / (n_min + n_maj)

    if current_ratio > r:
        # Je třeba ubrat minoritní třídu (nechat majoritní celou)
        target_min = int(round(r * n_maj / (1.0 - r)))
        target_min = max(1, min(target_min, n_min))

        keep_maj = idx_maj
        keep_min = rng.choice(idx_min, size=target_min, replace=False)
    else:
        # Je třeba ubrat majoritní třídu (nechat minoritní celou)
        target_maj = int(round(n_min * (1.0 - r) / r))
        target_maj = max(1, min(target_maj, n_maj))

        keep_min = idx_min
        keep_maj = rng.choice(idx_maj, size=target_maj, replace=False)

    keep = np.concatenate([keep_min, keep_maj])
    rng.shuffle(keep)

    return x[keep], y[keep]

def demonstrate_leakage(
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        random_state: int,
) -> dict:
    """Demonstruje unik dat: standardizace PRED vs. PO rozdeleni na train/test."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    test_size = 0.2

    # --- SPATNE: standardizace cele matice pred rozdelenim -----------------
    params_all = standardize_fit(x)
    x_all_std = standardize_apply(x, params_all)
    xtr_w, xte_w, ytr_w, yte_w = train_test_split(
        x_all_std, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf_w = KNeighborsClassifier(n_neighbors=k)
    clf_w.fit(xtr_w, ytr_w)
    acc_wrong = float(np.mean(clf_w.predict(xte_w) == yte_w))

    # --- SPRAVNE: nejdriv split, scaler jen z trenovacich dat -------------
    xtr_r, xte_r, ytr_r, yte_r = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    params_tr = standardize_fit(xtr_r)
    xtr_r_std = standardize_apply(xtr_r, params_tr)
    xte_r_std = standardize_apply(xte_r, params_tr)
    clf_r = KNeighborsClassifier(n_neighbors=k)
    clf_r.fit(xtr_r_std, ytr_r)
    acc_right = float(np.mean(clf_r.predict(xte_r_std) == yte_r))

    return {
        "acc_wrong": acc_wrong,
        "acc_right": acc_right,
        "test_size": test_size,
        "k": k,
    }
