import numpy as np

from gutenTAG.tsgen.capabilities.corrected_detectability_candidates import CandidateSpec
from gutenTAG.tsgen.capabilities.corrected_detectability_null_bundle import (
    clean_scan_score_rows,
)
from gutenTAG.tsgen.capabilities.scan_scores import CleanWindowScoreBlock


def test_clean_scan_score_rows_aligns_candidate_windows() -> None:
    candidates = (
        CandidateSpec("mean_z", (0,)),
        CandidateSpec("local_energy_z", (0,)),
    )
    block = CleanWindowScoreBlock(
        instance_keys=("instance",),
        window_counts=(3,),
        scores_by_witness={
            "mean_z": (np.asarray([1.0, np.nan, 3.0], dtype=np.float64),),
            "local_energy_z": (np.asarray([4.0, 5.0, 6.0], dtype=np.float64),),
        },
    )

    rows = clean_scan_score_rows(candidates, {(0,): block})

    assert rows == [
        {"mean_z@0": 1.0, "local_energy_z@0": 4.0},
        {"local_energy_z@0": 5.0},
        {"mean_z@0": 3.0, "local_energy_z@0": 6.0},
    ]
