import json
import logging

import pytest

from gutenTAG.tsgen.generation_run import (
    configure_generation_logging,
    prepare_output_directory,
    validate_non_empty_dataset,
    write_split_summary,
)


def test_prepare_output_directory_requires_overwrite_for_existing_root(
    tmp_path,
) -> None:
    output_root = tmp_path / "dataset"
    output_root.mkdir()
    (output_root / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_output_directory(output_root, overwrite_output=False)

    prepare_output_directory(output_root, overwrite_output=True)

    assert output_root.exists()
    assert not (output_root / "old.txt").exists()


def test_configure_generation_logging_writes_generation_log(tmp_path) -> None:
    logger = logging.getLogger("test.generation_run")

    configure_generation_logging(logger, output_root=tmp_path, log_level="INFO")
    logger.info("hello")

    log_text = (tmp_path / "generation.log").read_text(encoding="utf-8")
    assert "INFO | test.generation_run | hello" in log_text
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_validate_non_empty_dataset_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="produced no instances"):
        validate_non_empty_dataset(
            dataset_stats={"instance_count": 0},
            generated_variant_entries=(),
            allow_empty_dataset=False,
        )

    validate_non_empty_dataset(
        dataset_stats={"instance_count": 0},
        generated_variant_entries=(),
        allow_empty_dataset=True,
    )


def test_write_split_summary_writes_json(tmp_path) -> None:
    write_split_summary(
        split_dir=tmp_path,
        split="train",
        split_instance_summaries=[
            {
                "instance_id": "instance_000",
                "paired": True,
                "target_density": 0.25,
                "achieved_density": 0.25,
                "label_density": 0.25,
                "event_count": 1,
                "n_segments": 1,
                "segment_lengths": [5],
                "per_channel_segment_counts": {"0": 1, "1": 0},
            }
        ],
        length=20,
        channels=2,
    )

    payload = json.loads((tmp_path / "split_summary.json").read_text())
    assert payload["split"] == "train"
    assert payload["instances"] == 1
