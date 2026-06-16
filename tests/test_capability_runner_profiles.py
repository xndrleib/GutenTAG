from gutenTAG.tsgen.capabilities.runner_profiles import missing_required_tables


def test_missing_required_tables_reports_only_absent_keys() -> None:
    tables = {
        "event_summary": object(),
        "arity": object(),
        "implementation_validity": object(),
    }

    assert missing_required_tables(
        tables,
        (
            "event_summary",
            "arity",
            "implementation_validity",
            "detector_attribution",
            "corrected_detectability_frontier",
        ),
    ) == ["detector_attribution", "corrected_detectability_frontier"]


def test_missing_required_tables_accepts_complete_registry() -> None:
    tables = {"a": object(), "b": object()}

    assert missing_required_tables(tables, ("a", "b")) == []
