import pandas as pd

from gutenTAG.tsgen.capabilities.model_zoo_tables import (
    sort_event_scores,
    sort_frontier,
    sort_manifest,
    sort_table,
)


def test_sort_frontier_orders_by_event_model_and_alpha() -> None:
    frame = pd.DataFrame(
        [
            {"event_id": "e2", "model_id": "b", "alpha": 0.2},
            {"event_id": "e1", "model_id": "b", "alpha": 0.2},
            {"event_id": "e1", "model_id": "a", "alpha": 0.5},
            {"event_id": "e1", "model_id": "a", "alpha": 0.1},
        ]
    )

    sorted_frame = sort_frontier(frame)

    assert sorted_frame[["event_id", "model_id", "alpha"]].to_records(
        index=False
    ).tolist() == [
        ("e1", "a", 0.1),
        ("e1", "a", 0.5),
        ("e1", "b", 0.2),
        ("e2", "b", 0.2),
    ]


def test_sort_event_scores_and_manifest_ignore_missing_columns() -> None:
    event_scores = sort_event_scores(
        pd.DataFrame([{"event_id": "e2"}, {"event_id": "e1"}])
    )
    manifest = sort_manifest(
        pd.DataFrame([{"fit_variant_id": "v2"}, {"fit_variant_id": "v1"}])
    )

    assert event_scores["event_id"].tolist() == ["e1", "e2"]
    assert manifest["fit_variant_id"].tolist() == ["v1", "v2"]


def test_sort_table_returns_empty_frames_with_clean_index() -> None:
    frame = pd.DataFrame(columns=["event_id"])

    assert sort_table(frame, ("event_id",)).empty
    assert sort_table(frame, ("event_id",)).index.tolist() == []
