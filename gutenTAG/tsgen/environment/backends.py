"""Release backend policy checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Optional, Sequence

from packaging.version import Version

MIN_RELEASE_NEUROKIT2_VERSION = Version("0.2.13")


def validate_release_backend_policy(
    *,
    base_oscillations: Optional[Sequence[str]],
    all_base_oscillations: Sequence[str],
) -> None:
    """Validate release backend policy for selected carriers.

    ECG release datasets require a modern `neurokit2`; the historical
    deterministic fallback is not admitted for release-grade TS dataset
    generation.
    """
    selected = set(str(name) for name in (base_oscillations or all_base_oscillations))
    if "ecg" not in selected:
        return
    if find_spec("neurokit2") is None:
        raise RuntimeError(
            "ECG release dataset generation requires neurokit2>=0.2.13,<0.3. "
            "Install the locked release environment or remove 'ecg' from "
            "variants.base_oscillations."
        )
    try:
        installed_version = Version(version("neurokit2"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "ECG release dataset generation requires neurokit2>=0.2.13,<0.3. "
            "Install the locked release environment or remove 'ecg' from "
            "variants.base_oscillations."
        ) from exc
    if installed_version < MIN_RELEASE_NEUROKIT2_VERSION:
        raise RuntimeError(
            "ECG release dataset generation requires neurokit2>=0.2.13,<0.3; "
            f"found neurokit2=={installed_version}. Install the locked release "
            "environment or remove 'ecg' from variants.base_oscillations."
        )
