from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.distance import Distance


class KNNClassifier:
    """Induktivni klasifikator k nejblizsich sousedu (nededi z niceho)."""

    def __init__(self, k: int, distance: "Distance") -> None:
        self.k = k
        self.distance = distance
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.x_train_ = np.asarray(x, dtype=np.float64)
        self.y_train_ = np.asarray(y).ravel().astype(int)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.x_train_ is not None, "Model není nafitován (zavolejte nejprve fit)."
        assert x.shape[1] == self.x_train_.shape[1], "Počet příznaků dotazu neodpovídá trénovacím datům."

        y_pred = np.zeros(x.shape[0], dtype=int)

        for i, x_query in enumerate(x):
            # 1. Spočítání vzdálenosti ke všem trénovacím bodům
            distances = np.array([
                self.distance.calculate(x_query, x_train_i)
                for x_train_i in self.x_train_
            ])

            # 2. Výběr k nejbližších sousedů
            # kind='stable' zajistí zachování pořadí indexů v případě shody vzdáleností
            k_indices = np.argsort(distances, kind='stable')[:self.k]

            # 3. Získání popisků a většinový hlas
            k_labels = self.y_train_[k_indices]

            # np.bincount spočítá četnost jednotlivých tříd.
            # np.argmax vybere index (třídu) s nejvyšší četností.
            # Při shodě vrací první nalezené maximum, tj. třídu s nejnižším číslem.
            y_pred[i] = np.argmax(np.bincount(k_labels))

        return y_pred

    def save(self, path: str) -> None:
        assert self.x_train_ is not None, "Nelze uložit nenafitovaný model."

        np.savez(
            path,
            x_train=self.x_train_,
            y_train=self.y_train_,
            k=np.asarray(self.k)
        )

    def load(self, path: str) -> KNNClassifier:
        assert os.path.exists(path) or os.path.exists(path + '.npz'), f"Soubor {path} neexistuje."

        with np.load(path) as data:
            self.x_train_ = data['x_train']
            self.y_train_ = data['y_train']
            self.k = int(data['k'])

        return self
