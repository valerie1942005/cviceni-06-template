# -*- coding: utf-8 -*-

"""
Created on 01. 09. 2026 at 13:35:32

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
    
    Testy pro cviceni 06 — kNN a validace modelu / vyhodnoceni klasifikace.

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a jednotlive testy, ktere volaji nedokoncene
    ukoly, se oznaci jako xfail (ocekavane selhani s NotImplementedError) — nikdy
    neskonci holym tracebackem. Po dokonceni ukolu se z nich stanou xpass a
    nasledne plne prochazejici testy.

    Brana vzdalenosti (Distance) se v tomto cviceni VRACI: kNN skutecne vola
    ``distance.calculate`` pri hledani sousedu. Testy kNN proto pouzivaji
    nize definovanou tridu ``DummyDistance`` — plnohodnotnou eukleidovskou
    vzdalenost primo v testu — aby nezavisely na tom, zda student uz dokoncil
    ``src/distance.py``.

    Krome shody se ``sklearn`` testuje ``TestKNN`` i pravidlo pro remizu hlasu
    (nizsi trida) a validacni ``assert`` v ``predict`` (bez ``fit`` / neshoda
    poctu priznaku). ``TestKNNPersistence`` overuje round-trip ``save`` ->
    ``load`` (znovunacteny model dava stejne predikce a nese stejna naucena
    pole) — obdoba ``TestPCAPersistence`` z cviceni 05.

    Metriky se testuji proti ``sklearn`` (stejna orientace kontingecni tabulky
    ``[[TN, FP], [FN, TP]]``, pozitivni trida = maligni = 1); specificita, kterou
    sklearn primo nema, se overuje na rucne sestavene tabulce. Validacni
    strategie (``KFold``, ``Bootstrap``) se testuji na strukturalni vlastnosti
    (pokryti, neprekryv, losovani s opakovanim, out-of-bag komplement).
    Predzpracovani (``standardize_fit`` / ``standardize_apply``,
    ``subsample_imbalance``) se overuje proti ``sklearn`` a na dodrzeni
    ciloveho pomeru trid.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score
from sklearn.model_selection import KFold as SKKFold
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from dataio.preprocessing import (
    standardize_apply,
    standardize_fit,
    subsample_imbalance,
)
from src.distance import Distance
from src.knn import KNNClassifier
from src.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall_sensitivity,
    specificity,
)
from src.validation import Bootstrap, KFold, cross_validate

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="student ukol jeste neni dokoncen")


class DummyDistance(Distance):
    """Plnohodnotna eukleidovska (L2) vzdalenost pro ucely testu.

    Dedi ze ``src.distance.Distance`` a implementuje obe abstraktni casti
    (``is_metric`` i ``calculate``) konkretne — brana vzdalenosti je tedy
    v testech "otevrena" bez ohledu na stav ``src/distance.py``.
    """

    @property
    def is_metric(self) -> bool:
        """Eukleidovska vzdalenost je prava metrika."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Vrati eukleidovskou vzdalenost mezi vektory ``x`` a ``y``."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return float(np.sqrt(np.sum((x - y) ** 2)))


class _MajorityClassifier:
    """Trivialni model pro test ``cross_validate``: vzdy predikuje vetsinovou tridu."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_MajorityClassifier":
        """Zapamatuje si vetsinovou tridu z ``y``."""
        vals, counts = np.unique(np.asarray(y), return_counts=True)
        self._majority_ = vals[int(np.argmax(counts))]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Vrati pole delky ``len(x)`` vyplnene vetsinovou tridou z ``fit``."""
        return np.full(np.asarray(x).shape[0], self._majority_)


# --------------------------------------------------------------------------- #
#  Fixtury se synteticnymi daty (pevny seed)                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture
def separable_2class() -> tuple[np.ndarray, np.ndarray]:
    """Dobre oddelene dvojrozmerne shluky dvou trid (kNN je klasifikuje bezchybne)."""
    rng = np.random.default_rng(42)
    n = 60
    x0 = rng.normal(-2.0, 0.5, size=(n, 2))
    x1 = rng.normal(2.0, 0.5, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    return x, y


@pytest.fixture
def labels_pair() -> tuple[np.ndarray, np.ndarray]:
    """Dvojice ``(y_true, y_pred)`` z ``{0, 1}`` s obema tridami bohate zastoupenymi."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = y_true.copy()
    flip = rng.choice(200, size=45, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    return y_true, y_pred


def test_dummy_distance_brana_je_funkcni() -> None:
    """Kontrola, ze pomocna ``DummyDistance`` funguje i ve stavu stubu (neni ukol)."""
    d = DummyDistance()
    assert d.is_metric is True
    assert d.calculate(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)


class TestKNN:
    """Jedina studentska metoda: ``KNNClassifier.predict`` (vetsinovy hlas k sousedu)."""


    def test_predict_odpovida_sklearn_na_trenovacich_datech(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Kontrola, ze kNN predikuje stejne jako sklearn na trenovacich datech."""
        x, y = separable_2class
        k = 5
        mine = KNNClassifier(k, DummyDistance()).fit(x, y).predict(x)
        ref = SKKNeighborsClassifier(n_neighbors=k).fit(x, y).predict(x)
        assert np.array_equal(np.asarray(mine).ravel(), ref)


    def test_predict_odpovida_sklearn_na_novych_bodech(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shoda kNN se sklearn i na novych bodech mimo trenovaci mnozinu."""
        x, y = separable_2class
        k = 3
        rng = np.random.default_rng(7)
        x_query = np.vstack([
            rng.normal(-2.0, 0.5, size=(15, 2)),
            rng.normal(2.0, 0.5, size=(15, 2)),
        ])
        mine = KNNClassifier(k, DummyDistance()).fit(x, y).predict(x_query)
        ref = SKKNeighborsClassifier(n_neighbors=k).fit(x, y).predict(x_query)
        assert np.array_equal(np.asarray(mine).ravel(), ref)


    def test_predict_resi_shodu_hlasu_nizsi_tridou(self) -> None:
        """Pri remize hlasu vraci trida s NEJNIZSIM cislem (pravidlo z docstringu predict).

        Dva trenovaci body ruznych trid, dotaz presne uprostred, ``k = 2`` ->
        jeden hlas pro tridu 0, jeden pro tridu 1 -> remiza -> ocekava se 0.
        """
        x_train = np.array([[0.0, 0.0], [10.0, 10.0]])
        y_train = np.array([0, 1])
        q = np.array([[5.0, 5.0]])
        pred = KNNClassifier(2, DummyDistance()).fit(x_train, y_train).predict(q)
        assert int(np.asarray(pred).ravel()[0]) == 0


    def test_predict_bez_fit_vyvola_chybu(self) -> None:
        """predict pred zavolanim fit musi selhat (assert v kostre), ne vratit nesmysl."""
        knn = KNNClassifier(3, DummyDistance())
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            knn.predict(np.zeros((2, 2)))


    def test_predict_neshoda_poctu_priznaku_vyvola_chybu(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Dotaz s jinym poctem priznaku nez trenovaci data musi selhat (assert)."""
        x, y = separable_2class          # 2 priznaky
        knn = KNNClassifier(3, DummyDistance()).fit(x, y)
        with pytest.raises((AssertionError, ValueError, IndexError)):
            knn.predict(np.zeros((4, 5)))   # 5 priznaku


class TestKNNPersistence:
    """Model = ulozena trenovaci mnozina:
        save -> load round-trip (jako TestPCAPersistence v cv5)."""


    def test_save_load_zachova_predikce(
        self, separable_2class: tuple[np.ndarray, np.ndarray], tmp_path
    ) -> None:
        """Po ``save`` -> ``load`` dava znovunacteny model stejne predikce jako original."""
        x, y = separable_2class
        knn = KNNClassifier(5, DummyDistance()).fit(x, y)
        pred = np.asarray(knn.predict(x)).ravel()
        cesta = tmp_path / "knn_model.npz"
        knn.save(str(cesta))
        assert cesta.exists()
        obnovena = KNNClassifier(5, DummyDistance()).load(str(cesta))
        assert np.array_equal(np.asarray(obnovena.predict(x)).ravel(), pred)


    def test_save_load_zachova_naucena_pole(
        self, separable_2class: tuple[np.ndarray, np.ndarray], tmp_path
    ) -> None:
        """Po ``save`` -> ``load`` sedi naucena pole a ``k`` se bere ze souboru."""
        x, y = separable_2class
        knn = KNNClassifier(7, DummyDistance()).fit(x, y)
        cesta = tmp_path / "knn_model.npz"
        knn.save(str(cesta))
        # k se ma nacist ZE SOUBORU, ne z konstruktoru (zde schvalne jine)
        obnovena = KNNClassifier(1, DummyDistance()).load(str(cesta))
        assert np.allclose(np.asarray(obnovena.x_train_), np.asarray(knn.x_train_))
        assert np.array_equal(np.asarray(obnovena.y_train_).ravel(),
                              np.asarray(knn.y_train_).ravel())
        assert int(obnovena.k) == 7


class TestMetrics:
    """Kontingecni tabulka jako koren; vsechny metriky se z ni odvozuji."""


    def test_confusion_matrix_odpovida_sklearn(
        self, labels_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Tvar (2, 2) a shoda se ``sklearn`` (orientace [[TN, FP], [FN, TP]])."""
        y_true, y_pred = labels_pair
        cm = np.asarray(confusion_matrix(y_true, y_pred))
        assert cm.shape == (2, 2)
        assert np.array_equal(cm, sk_confusion_matrix(y_true, y_pred, labels=[0, 1]))


    def test_accuracy_odpovida_sklearn(
        self, labels_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """``accuracy(cm)`` se shoduje s ``sklearn.metrics.accuracy_score``."""
        y_true, y_pred = labels_pair
        cm = confusion_matrix(y_true, y_pred)
        assert accuracy(cm) == pytest.approx(sk_accuracy_score(y_true, y_pred))


    def test_precision_odpovida_sklearn(
        self, labels_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """``precision(cm)`` se shoduje s ``sklearn.metrics.precision_score``."""
        y_true, y_pred = labels_pair
        cm = confusion_matrix(y_true, y_pred)
        assert precision(cm) == pytest.approx(
            sk_precision_score(y_true, y_pred, zero_division=0)
        )


    def test_recall_sensitivity_odpovida_sklearn(
        self, labels_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """``recall_sensitivity(cm)`` se shoduje s ``sklearn.metrics.recall_score``."""
        y_true, y_pred = labels_pair
        cm = confusion_matrix(y_true, y_pred)
        assert recall_sensitivity(cm) == pytest.approx(
            sk_recall_score(y_true, y_pred, zero_division=0)
        )


    def test_f1_odpovida_sklearn(
        self, labels_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """``f1_score(cm)`` se shoduje s ``sklearn.metrics.f1_score``."""
        y_true, y_pred = labels_pair
        cm = confusion_matrix(y_true, y_pred)
        assert f1_score(cm) == pytest.approx(
            sk_f1_score(y_true, y_pred, zero_division=0)
        )


    def test_specificity_na_rucni_tabulce(self) -> None:
        """Specificitu sklearn primo nema — overujeme na rucne sestavene tabulce.

        ``cm = [[TN, FP], [FN, TP]] = [[80, 20], [10, 90]]``
        ``Sp = TN / (TN + FP) = 80 / 100 = 0.8``.
        """
        cm = np.array([[80, 20], [10, 90]])
        assert specificity(cm) == pytest.approx(0.8)
        # hranicni pripad: zadny negativni vzorek -> jmenovatel 0 -> 0.0
        assert specificity(np.array([[0, 0], [5, 7]])) == pytest.approx(0.0)


class TestValidation:
    """Prevzorkovaci strategie: ``KFold`` (pokryti/neprekryv) a ``Bootstrap`` (OOB)."""


    def test_kfold_pokryti_a_neprekryv(self) -> None:
        """Foldy se neprekryvaji, pokryji vsechny indexy a maji velikosti jako sklearn."""
        n, k = 23, 5
        x = np.zeros((n, 2))
        y = np.zeros(n)
        folds = list(KFold(k, shuffle=True, random_state=0).split(x, y))

        assert len(folds) == k
        test_parts = [np.asarray(te) for _, te in folds]

        # kazdy vzorek je testovan prave jednou
        all_test = np.concatenate(test_parts)
        assert np.array_equal(np.sort(all_test), np.arange(n))

        for tr, te in folds:
            tr = np.asarray(tr)
            te = np.asarray(te)
            assert set(tr.tolist()).isdisjoint(set(te.tolist()))
            assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(n))

        # struktura (velikosti foldu) odpovida sklearn.model_selection.KFold
        sk_sizes = sorted(
            len(te) for _, te in SKKFold(n_splits=k, shuffle=True, random_state=0).split(x)
        )
        assert sorted(len(te) for te in test_parts) == sk_sizes


    def test_bootstrap_opakovani_a_oob_komplement(self) -> None:
        """Trenink ma delku ``n`` s opakovanim, OOB test je komplement unikatnich indexu."""
        n, b = 40, 10
        x = np.zeros((n, 2))
        y = np.zeros(n)
        splits = list(Bootstrap(b, random_state=1).split(x, y))

        assert 1 <= len(splits) <= b
        for tr, te in splits:
            tr = np.asarray(tr)
            te = np.asarray(te)
            # trenovaci mnozina ma delku n a je losovana S OPAKOVANIM
            assert tr.shape[0] == n
            assert np.unique(tr).size < n
            # out-of-bag test = presny komplement unikatnich trenovacich indexu
            oob = np.setdiff1d(np.arange(n), np.unique(tr))
            assert np.array_equal(np.sort(te), oob)
            assert set(np.unique(tr).tolist()).isdisjoint(set(te.tolist()))


    def test_cross_validate_vraci_dohodnute_klice(self) -> None:
        """Vysledek ma klice ``*_mean`` / ``*_std``, ``per_split_scores`` a ``n_splits``."""
        rng = np.random.default_rng(0)
        n = 50
        x = rng.normal(size=(n, 3))
        y = rng.integers(0, 2, size=n)
        res = cross_validate(
            _MajorityClassifier(),
            KFold(5, random_state=0),
            x, y,
            {"accuracy": accuracy, "recall_Se": recall_sensitivity},
        )
        for key in ("accuracy_mean", "accuracy_std", "recall_Se_mean", "recall_Se_std"):
            assert key in res
        assert set(res["per_split_scores"]) == {"accuracy", "recall_Se"}
        assert res["n_splits"] == 5
        assert len(res["per_split_scores"]["accuracy"]) == 5


class TestPreprocessing:
    """Standardizace rozdelena na fit/apply (obrana proti uniku dat) a
    podvzorkovani pro demonstraci nevyvazenosti."""


    def test_standardize_fit_parametry_odpovidaji_sklearn(self) -> None:
        """``standardize_fit`` vraci ``mean`` a ``std`` shodne se ``StandardScaler``."""
        rng = np.random.default_rng(3)
        x_train = rng.normal(5.0, 2.0, size=(50, 4))
        params = standardize_fit(x_train)
        sk = SKStandardScaler().fit(x_train)
        assert np.allclose(np.asarray(params["mean"]), sk.mean_)
        assert np.allclose(np.asarray(params["std"]), sk.scale_)


    def test_standardize_apply_odpovida_sklearn(self) -> None:
        """``standardize_apply`` na train i test odpovida ``StandardScaler.transform``."""
        rng = np.random.default_rng(4)
        x_train = rng.normal(5.0, 2.0, size=(50, 4))
        x_test = rng.normal(5.0, 2.0, size=(20, 4))
        params = standardize_fit(x_train)
        sk = SKStandardScaler().fit(x_train)
        assert np.allclose(np.asarray(standardize_apply(x_train, params)),
                           sk.transform(x_train))
        assert np.allclose(np.asarray(standardize_apply(x_test, params)),
                           sk.transform(x_test))


    def test_standardize_apply_konstantni_sloupec_bez_nan(self) -> None:
        """Konstantni sloupec (std = 0) se jen vycentruje na nuly, bez inf/nan."""
        x_train = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0], [4.0, 7.0]])
        params = standardize_fit(x_train)
        out = np.asarray(standardize_apply(x_train, params))
        assert np.all(np.isfinite(out))
        # konstantni sloupec se jen vycentruje -> vysledek jsou same nuly
        assert np.allclose(out[:, 1], 0.0)


    def test_subsample_imbalance_dodrzi_cilovy_podil(self) -> None:
        """Dodrzi cilovy podil minoritni tridy a vrati obe tridy neprazdne."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=(300, 3))
        y = np.array([0] * 150 + [1] * 150)
        x_imb, y_imb = subsample_imbalance(x, y, minority_ratio=0.1, random_state=0)
        y_imb = np.asarray(y_imb).astype(int)

        assert np.asarray(x_imb).shape[0] == y_imb.shape[0]
        assert y_imb.shape[0] > 0
        assert set(np.unique(y_imb).tolist()) == {0, 1}
        minority_frac = np.bincount(y_imb).min() / y_imb.shape[0]
        assert minority_frac == pytest.approx(0.1, abs=0.03)
