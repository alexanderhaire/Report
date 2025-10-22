import pandas as pd
from pandas.testing import assert_series_equal

from helloworld import _build_features


def test_horizon_two_days_produces_two_step_target():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "chem_X": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "fut_F": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        },
        index=idx,
    )

    X, y = _build_features(df, "X", lags=2, horizon="2D")

    expected = df["chem_X"].shift(-2).loc[y.index].rename("y")

    assert_series_equal(y, expected)
    assert list(X.index) == list(y.index)
