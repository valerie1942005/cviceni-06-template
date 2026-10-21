# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
International Clinical Research Center, Brno
Brno University of Technology, Brno
GitHub: RicRedi

(._.)
<|>
_/|_

Description:
Abstraktní třída vzdálenosti a její konkrétní implementace.
"""

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """Abstraktní základ pro metriky vzdálenosti.

    Každá konkrétní metrika dědí od této třídy a implementuje metodu
    ``calculate``. Metoda ``create_distance_matrix`` je sdílená a volá
    ``calculate`` v cyklu — není třeba ji přepisovat.
    """

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí ``True``, pokud vzdálenost splňuje axiomy metriky."""
        pass

    @abstractmethod
    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá vzdálenost mezi dvěma body.

        Parameters
        ----------
        point_a:
            První bod — pole tvaru ``(n_příznaků,)``.
        point_b:
            Druhý bod — pole tvaru ``(n_příznaků,)``.

        Returns
        -------
        float
            Vzdálenost mezi ``point_a`` a ``point_b``.
        """
        assert point_a.ndim == 1 and point_b.ndim == 1, "Vstupy musí být 1D vektory."
        assert point_a.shape[0] == point_b.shape[0], "Vektory musí mít stejnou délku."
        pass

    def create_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Vytvoří čtvercovou matici vzdáleností mezi všemi dvojicemi bodů.

        Matice je symetrická s nulovou diagonálou.
        Volá ``self.calculate`` pro každou dvojici — implementace metriky
        není třeba zde duplikovat.

        Parameters
        ----------
        data:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Matice vzdáleností tvaru ``(n_bodů, n_bodů)``.
        """
        assert data.ndim == 2, "Data musí být 2D matice."
        assert data.shape[0] >= 2, "Matice musí obsahovat alespoň 2 body."

        n: int = data.shape[0]
        matrix: np.ndarray = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist: float = self.calculate(data[i], data[j])
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix


class EuclideanDistance(Distance):
    """Euklidovská vzdálenost — délka přímé spojnice dvou bodů."""

    @property
    def is_metric(self) -> bool:
        """Euklidovská vzdálenost je pravá metrika."""
        return True

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá euklidovskou vzdálenost mezi dvěma body.

        Parameters
        ----------
        point_a:
            První bod.
        point_b:
            Druhý bod.

        Returns
        -------
        float
            Euklidovská vzdálenost.
        """
        assert point_a.ndim == 1 and point_b.ndim == 1, "Vstupy musí být 1D vektory."
        assert point_a.shape[0] == point_b.shape[0], "Vektory musí mít stejnou délku."

        return float(np.sqrt(np.sum((point_a - point_b) ** 2)))


class ManhattanDistance(Distance):
    """Manhattanská vzdálenost — součet absolutních rozdílů souřadnic."""

    @property
    def is_metric(self) -> bool:
        """Manhattanská vzdálenost je pravá metrika."""
        return True

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá manhattanskou vzdálenost mezi dvěma body.

        Parameters
        ----------
        point_a:
            První bod.
        point_b:
            Druhý bod.

        Returns
        -------
        float
            Manhattanská vzdálenost.
        """
        assert point_a.ndim == 1 and point_b.ndim == 1, "Vstupy musí být 1D vektory."
        assert point_a.shape[0] == point_b.shape[0], "Vektory musí mít stejnou délku."

        return float(np.sum(np.abs(point_a - point_b)))


class CosineCoeficient(Distance):
    """Kosinová podobnost (jako vzdálenost: 1 - kosinová_podobnost)."""

    @property
    def is_metric(self) -> bool:
        """Kosinová vzdálenost není pravá metrika (porušuje trojúhelníkovou nerovnost)."""
        return False

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá kosinovou vzdálenost mezi dvěma body.

        Parameters
        ----------
        point_a:
            První bod.
        point_b:
            Druhý bod.

        Returns
        -------
        float
            Kosinová vzdálenost v rozsahu [0, 2].
        """
        assert point_a.ndim == 1 and point_b.ndim == 1, "Vstupy musí být 1D vektory."
        assert point_a.shape[0] == point_b.shape[0], "Vektory musí mít stejnou délku."

        norm_a = np.linalg.norm(point_a)
        norm_b = np.linalg.norm(point_b)

        # Ošetření nulové normy (nulového vektoru)
        if norm_a == 0 or norm_b == 0:
            return 1.0  # Znamená to, že vektory nejsou podobné

        cosine_similarity = np.dot(point_a, point_b) / (norm_a * norm_b)
        return float(1 - cosine_similarity)
