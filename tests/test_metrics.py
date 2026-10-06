import numpy as np
import pytest

from voicesent.metrics import intensity_label, mosi_metrics


def test_metrics_hand_computed():
    true = np.array([2.0, -1.0, 0.0, 3.0, -2.6])
    pred = np.array([1.0, -0.4, 0.3, -0.2, -3.0])
    m = mosi_metrics(pred, true)
    assert m["mae"] == pytest.approx(np.mean([1, .6, .3, 3.2, .4]))
    assert m["acc2_nonzero"] == pytest.approx(3 / 4)      # neutral excluded; 3.0 vs -0.2 wrong
    assert m["acc2_has0"] == pytest.approx(4 / 5)
    assert m["acc7"] == pytest.approx(2 / 5)              # (-1 vs -0) no, (0,0) yes, (-3,-3) yes


def test_intensity():
    assert intensity_label(2.6) == "strongly positive"
    assert intensity_label(-0.4) == "neutral"
    assert intensity_label(-9) == "strongly negative"
