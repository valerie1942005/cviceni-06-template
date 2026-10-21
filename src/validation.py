from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from src.metrics import confusion_matrix


class Validator(ABC):
    """Spolecne rozhrani vsech prevzorkovacich strategii (navrhovy vzor Strategy)."""

    @abstractmethod
    def split(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje dvojice (train_idx, test_idx) celociselnych indexu."""


class HoldOut(Validator):
    """Jedno nahodne rozdeleni na train/test (nejjednodussi prevzorkovani)."""

    def __init__(
            self,
            test_size: float = 0.2,
            shuffle: bool = True,
            random_state: int | None = None,
    ) -> None:
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
            self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            rng.shuffle(indices)

        n_test = int(round(self.test_size * n_samples))
        n_test = max(1, min(n_test, n_samples - 1))
        n_train = n_samples - n_test

        train_idx = np.sort(indices[:n_train])
        test_idx = np.sort(indices[n_train:])
        yield train_idx, test_idx


class KFold(Validator):
    """k-nasobna krizova validace — n_folds rovnocennych rozdeleni."""

    def __init__(
            self,
            n_folds: int,
            shuffle: bool = True,
            random_state: int | None = None,
    ) -> None:
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
            self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = x.shape[0]

        assert self.n_folds >= 2, "KFold vyžaduje alespoň 2 foldy."
        assert self.n_folds <= n_samples, "Počet foldů nemůže být větší než počet vzorků."

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        # np.array_split rozdelí pole na n_folds částí i když není n_samples dělitelné beze zbytku
        folds = np.array_split(indices, self.n_folds)

        for i in range(self.n_folds):
            # Testovací množina je i-tý fold
            test_idx = folds[i]

            # Trénovací množina vznikne spojením všech OSTATNÍCH foldů
            train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != i])

            # Obě pole yieldneme vzestupně seřazená
            yield np.sort(train_idx), np.sort(test_idx)


class Bootstrap(Validator):
    """Bootstrap — vyber trenovaci mnoziny losovanim s opakovanim."""

    def __init__(
            self,
            n_bootstrap: int,
            random_state: int | None = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def split(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = x.shape[0]

        assert self.n_bootstrap >= 1, "Musí být provedena alespoň jedna iterace bootstrapu."
        assert n_samples > 0, "Pole vzorků (X) nesmí být prázdné."

        # Inicializace jediného generátoru před smyčkou pro zaručení nezávislosti iterací
        rng = np.random.default_rng(self.random_state)
        all_indices = np.arange(n_samples)

        for _ in range(self.n_bootstrap):
            # a. n indexů losovaných S OPAKOVÁNÍM
            train_idx = rng.choice(n_samples, size=n_samples, replace=True)

            # b. Out-of-bag (OOB) indexy. np.setdiff1d automaticky vrátí unikátní a seřazené hodnoty.
            test_idx = np.setdiff1d(all_indices, train_idx)

            # c. Iterace se přeskočí, pokud test_idx vyjde prázdné (vylosovaly se naprosto všechny indexy)
            if test_idx.size == 0:
                continue

            # d. Yield výsledku
            yield train_idx, test_idx


def cross_validate(
        model: Any,
        validator: Validator,
        x: np.ndarray,
        y: np.ndarray,
        metric_fns: dict[str, Callable[[np.ndarray], float]],
) -> dict:
    """Spusti model na vsech splitech dane strategie a agreguje metriky."""
    per_split_scores: dict[str, list[float]] = {name: [] for name in metric_fns}
    n_splits = 0

    for train_idx, test_idx in validator.split(x, y):
        n_splits += 1
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        cm = confusion_matrix(y[test_idx], y_pred)
        for name, fn in metric_fns.items():
            per_split_scores[name].append(float(fn(cm)))

    result: dict = {}
    for name, scores in per_split_scores.items():
        arr = np.asarray(scores, dtype=float)
        result[f"{name}_mean"] = float(arr.mean()) if arr.size else float("nan")
        result[f"{name}_std"] = float(arr.std()) if arr.size else float("nan")
    result["per_split_scores"] = per_split_scores
    result["n_splits"] = n_splits
    return result
