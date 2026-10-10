import numpy as np
import pytest
from eeg_ml.algorithms import patient_split, make_pipeline, fit_training_partition, cross_validation_indices


def table():
    rng = np.random.default_rng(7)
    return rng.normal(size=(32, 3)), np.tile([0, 1], 16), np.repeat(np.arange(8), 4)


def test_group_split_is_disjoint_and_reproducible():
    x, y, g = table()
    a, b = patient_split(x, y, g)
    assert not set(g[a]) & set(g[b])
    assert sorted(np.r_[a, b].tolist()) == list(range(len(x)))
    np.testing.assert_array_equal(a, patient_split(x, y, g)[0])


def test_scaler_only_observes_training_rows():
    x, y, g = table()
    train, test = patient_split(x, y, g)
    fitted, heldout = fit_training_partition(x, y, g, model="logistic")
    np.testing.assert_allclose(fitted["scale"].mean_, x[train].mean(axis=0))
    np.testing.assert_array_equal(test, heldout)
    assert fitted["scale"].n_samples_seen_ == len(train)


@pytest.mark.parametrize("model", ["svm", "logistic", "forest"])
def test_pipeline_is_unfitted(model):
    assert not hasattr(make_pipeline(model)["scale"], "mean_")


def test_cross_validation_has_no_patient_leakage():
    x, y, g = table()
    for train, test in cross_validation_indices(x, y, g, 4):
        assert not set(g[train]) & set(g[test])


def test_single_group_rejected():
    x, y, g = table()
    with pytest.raises(ValueError):
        patient_split(x, y, np.zeros_like(g))
