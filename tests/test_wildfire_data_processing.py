import numpy as np
import pandas as pd
import pytest

from test_scripts.validation import validate_output

REQUIRED_COLS = ["emc", "kbdi", "drought_factor", "ffdi",
                  "rate_of_spread", "kbdi_spinup_flag", "label"]


def make_clean_df(n=5):
    return pd.DataFrame({
        "emc": np.full(n, 10.0),
        "kbdi": np.full(n, 50.0),
        "drought_factor": np.full(n, 5.0),
        "ffdi": np.full(n, 12.0),
        "rate_of_spread": np.full(n, 0.5),
        "kbdi_spinup_flag": [False] * n,
        "label": [0, 1] * (n // 2) + [0] * (n % 2),
    })


def test_validate_output_passes_clean_data():
    df = make_clean_df()
    result = validate_output(df, raise_threshold=float("inf"))
    assert sum(result.values()) == 0


def test_validate_output_raises_on_missing_columns():
    df = make_clean_df().drop(columns=["ffdi"])
    with pytest.raises(AssertionError):
        validate_output(df)


def test_validate_output_flags_out_of_range_emc():
    df = make_clean_df()
    df.loc[0, "emc"] = 999
    result = validate_output(df, raise_threshold=float("inf"))
    assert result["emc"] == 1


def test_validate_output_flags_negative_ffdi():
    df = make_clean_df()
    df.loc[0, "ffdi"] = -1
    result = validate_output(df, raise_threshold=float("inf"))
    assert result["ffdi"] == 1


def test_validate_output_flags_nulls():
    df = make_clean_df()
    df.loc[0, "kbdi"] = np.nan
    result = validate_output(df, raise_threshold=float("inf"))
    assert result["nulls"] >= 1


def test_validate_output_raises_by_default_on_any_violation():
    df = make_clean_df()
    df.loc[0, "drought_factor"] = 999
    with pytest.raises(ValueError):
        validate_output(df)


def test_validate_output_respects_custom_threshold():
    df = make_clean_df()
    df.loc[0, "kbdi"] = 999
    result = validate_output(df, raise_threshold=1)
    assert result["kbdi"] == 1