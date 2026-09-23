"""Streaming benchmark writer; oracle artifacts never occupy detector inputs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import BenchmarkConfig
from ..processes.laws import sample_law
from ..processes.interventions import Intervention, apply_interventions, RELATION_ONLY
from ..sensor_artifacts import SENSOR_KINDS

LOGGER = logging.getLogger(__name__)


def derive_seed(seed: int, *parts: str) -> int:
    """128-bit, unambiguously namespaced v13 seeds; legacy seeds are unchanged."""
    payload = json.dumps([int(seed), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def write_json(path: Path, value: Any) -> None:
    """Write strict canonical JSON (no self-referential checksums)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compatible(family: str, mechanism: str, channels: int) -> bool:
    """Admit only implemented combinations, without fallback rewrites."""
    if mechanism in RELATION_ONLY:
        return family == "diagonal-ar" and channels >= 2
    if mechanism == "factor-loading":
        return family == "gp-lmc"
    return True


def generate_benchmark(
    config: BenchmarkConfig, output: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Construct fixed laws, independent replicas and paired query interventions.

    Each target system has its own independent clean reference and calibration
    trajectories. Calibration is not the clean counterfactual of any query.
    One event per query is intentional: repeated queries are independent
    realizations, not pseudoreplicated windows from one long trajectory.
    """
    output = Path(output).resolve()
    if output.exists():
        marker = output / ".synthgen-v13-owned"
        if (
            not overwrite
            or not marker.is_file()
            or marker.read_text() != "synthgen.v13.1\n"
        ):
            raise FileExistsError("Refusing to replace existing/unowned output")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        manifest = _generate(config, stage)
        (stage / ".synthgen-v13-owned").write_text("synthgen.v13.1\n")
        if output.exists():
            shutil.rmtree(output)
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage)
        raise
    return manifest


def _generate(config: BenchmarkConfig, root: Path) -> dict[str, Any]:
    systems = []
    identities: set[str] = set()
    skipped = []
    for split, spec in config.splits.items():
        LOGGER.info(
            "Generating split=%s role=%s systems=%d", split, spec.role, spec.systems
        )
        for index in tqdm(range(spec.systems), desc=split, leave=False):
            law_seed = derive_seed(config.master_seed, "law", split, str(index))
            rng = np.random.default_rng(law_seed)
            family = spec.families[index % len(spec.families)]
            channels = spec.channels[(index // len(spec.families)) % len(spec.channels)]
            params = {
                "length_scale": float(np.exp(rng.uniform(*np.log(spec.length_scale)))),
                "memory": float(rng.uniform(*spec.memory)),
                "noise_std": float(rng.uniform(*spec.noise_std)),
            }
            law = sample_law(
                family,
                channels,
                law_seed,
                rank=config.rank,
                features=config.features,
                **params,
            )
            if law.law_id in identities:
                raise ValueError("Same law occurs in two split/system slots")
            identities.add(law.law_id)
            system_id = law.law_id.split(":")[1]
            folder = root / "splits" / split / system_id
            (folder / "queries").mkdir(parents=True)
            (folder / "oracle").mkdir()
            (folder / "calibration").mkdir()
            write_json(root / "laws" / f"{system_id}.json", law.to_dict())
            ref_seed = derive_seed(law_seed, "reference")
            reference = law.sample(config.reference_length, ref_seed)
            np.savez_compressed(folder / "reference.npz", values=reference.values)
            calibration_seeds = []
            for k in range(config.calibration_realizations):
                seed = derive_seed(law_seed, "calibration", str(k))
                calibration_seeds.append(seed)
                normal = law.sample(config.length, seed)
                np.savez_compressed(
                    folder / "calibration" / f"{k:05d}.npz", values=normal.values
                )
            plan_rng = np.random.default_rng(derive_seed(law_seed, "plan"))
            length = max(
                4, int(config.length * plan_rng.uniform(*config.event_fraction))
            )
            start = int(plan_rng.integers(1, config.length - length))
            transition = min(
                (length - 1) // 2,
                int(length * plan_rng.uniform(*config.transition_fraction)),
            )
            strength = float(plan_rng.uniform(*config.strength))
            channel = int(plan_rng.integers(channels))
            query_records = []
            if spec.role != "calibration":
                for mechanism in spec.mechanisms:
                    if not compatible(family, mechanism, channels):
                        skipped.append(
                            {
                                "split": split,
                                "family": family,
                                "mechanism": mechanism,
                                "reason": "unsupported law/operator contract",
                            }
                        )
                        continue
                    signed = strength
                    if (
                        mechanism not in RELATION_ONLY + SENSOR_KINDS
                        and plan_rng.random() < 0.5
                    ):
                        signed = -signed
                    event = Intervention(
                        mechanism,
                        start,
                        start + length,
                        0.0 if mechanism == "null" else signed,
                        (channel,),
                        transition,
                    )
                    genotype = hashlib.sha256(
                        json.dumps(
                            {"law_id": law.law_id, **vars(event)}, sort_keys=True
                        ).encode()
                    ).hexdigest()
                    for replica in range(config.replicas_per_genotype):
                        seed = derive_seed(law_seed, "query", mechanism, str(replica))
                        clean = law.sample(config.length, seed)
                        changed = apply_interventions(
                            clean, [event], seed=derive_seed(seed, "observation")
                        )
                        key = f"{mechanism}-{replica:05d}"
                        np.savez_compressed(
                            folder / "queries" / f"{key}.npz",
                            values=changed.values,
                            observed_mask=changed.observed_mask,
                            timestamps=changed.timestamps,
                        )
                        np.savez_compressed(
                            folder / "oracle" / f"{key}.npz",
                            clean=clean.values,
                            intervention_mask=changed.intervention_mask,
                            effect_mask=changed.effect_mask,
                            evaluation_mask=changed.evaluation_mask,
                        )
                        write_json(folder / "oracle" / f"{key}.json", changed.events)
                        query_records.append(
                            {
                                "id": key,
                                "realization_id": clean.realization_id,
                                "genotype_id": genotype,
                                "seed": seed,
                                "mechanism": mechanism,
                                "replica": replica,
                            }
                        )
            record = {
                "system_id": system_id,
                "law_id": law.law_id,
                "split": split,
                "role": spec.role,
                "family": family,
                "channels": channels,
                "parameters": params,
                "law_seed": law_seed,
                "reference_seed": ref_seed,
                "calibration_seeds": calibration_seeds,
                "queries": query_records,
                "relative_path": str(folder.relative_to(root)),
            }
            all_seeds = [
                ref_seed,
                *calibration_seeds,
                *(q["seed"] for q in query_records),
            ]
            if len(set(all_seeds)) != len(all_seeds):
                raise ValueError("seed collision across reference/calibration/query")
            write_json(folder / "system.json", record)
            systems.append(record)
    hashes = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema_version": config.schema_version,
        "config": config.model_dump(mode="json"),
        "systems": systems,
        "skipped_combinations": skipped,
        "sha256": hashes,
        "status": "experimental_not_empirically_admitted",
        "calibration_unit": "independent_trajectory_from_same_target_law",
        "test_oracle_forbidden_during_scoring": True,
    }
    write_json(root / "dataset_manifest.json", manifest)
    LOGGER.info(
        "Generation complete: systems=%d queries=%d",
        len(systems),
        sum(len(s["queries"]) for s in systems),
    )
    return manifest
