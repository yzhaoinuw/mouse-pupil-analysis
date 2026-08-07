import numpy as np

from make_gif import diagnostic_segment


def test_diagnostic_segment_keeps_rejected_run_and_neighboring_endpoints():
    values = np.arange(7, dtype=float)
    rejected = np.array([False, False, True, True, False, False, False])

    result = diagnostic_segment(values, rejected)

    np.testing.assert_allclose(
        result,
        np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan]),
        equal_nan=True,
    )
