import pandas as pd

from gutenTAG.tsgen.capabilities.visualization.audit_index import (
    BUCKETS,
    build_html_index,
    build_manifest,
    build_markdown_index,
)


def test_visual_audit_index_renderers_include_panels_and_escape_html() -> None:
    selection = pd.DataFrame(
        [
            {
                "bucket": "strongest",
                "event_id": "event<1>",
                "variant_id": "variant&1",
                "admission_status": "release_with_warning",
                "warning_reasons": "shortcut<risk>",
                "plot_path": "strongest/event.png",
                "relation_scatter_path": "strongest/event__scatter.png",
                "rolling_correlation_path": "",
                "pca_residual_path": "strongest/event__pca.png",
            }
        ]
    )

    markdown = build_markdown_index(selection)
    html = build_html_index(selection)
    manifest = build_manifest(selection)

    assert "## strongest" in markdown
    assert "![event<1>](strongest/event.png)" in markdown
    assert "relation scatter" in markdown
    assert "PCA residual" in markdown
    assert "event&lt;1&gt;" in html
    assert "variant&amp;1" in html
    assert "shortcut&lt;risk&gt;" in html
    assert "<section><h2>strongest</h2>" in html
    assert manifest["selected_event_count"] == 1
    assert manifest["bucket_counts"] == {"strongest": 1}
    assert manifest["buckets"] == list(BUCKETS)
    assert manifest["relation_scatter_count"] == 1
    assert manifest["rolling_correlation_count"] == 0
    assert manifest["pca_residual_count"] == 1


def test_visual_audit_index_renderers_handle_empty_selection() -> None:
    selection = pd.DataFrame()

    assert "No visual audit events were selected." in build_markdown_index(selection)
    assert "No visual audit events were selected." in build_html_index(selection)
    manifest = build_manifest(selection)
    assert manifest["selected_event_count"] == 0
    assert manifest["bucket_counts"] == {}
