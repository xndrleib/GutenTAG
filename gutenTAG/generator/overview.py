import os
from typing import List, Dict, Optional, Any, Callable, Union

import numpy as np
import yaml

try:
    import git
except ModuleNotFoundError:
    git = None  # type: ignore[assignment]


class DictSanitizer:
    NUMPY_TYPES = (np.generic, np.ndarray)

    def sanitize(self, obj: Dict) -> Dict:
        for key, value in obj.items():
            obj[key] = self._sanitize_value(value)
        return obj

    def _decode_numpy_types(self, obj: Any) -> Any:
        """
        Taken from [numpyencoder](https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py)
        """

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.generic):
            scalar = obj.item()
            if isinstance(scalar, (bytes, bytearray, memoryview)):
                return None
            if isinstance(scalar, complex):
                return {"real": scalar.real, "imag": scalar.imag}
            return scalar

    def _sanitize_value(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            obj = self.sanitize(obj)  # type: ignore # recursive typecheck
        elif isinstance(obj, list):
            obj = list(map(self._sanitize_value, obj))  # type: ignore # recursive typecheck
        elif isinstance(obj, self.NUMPY_TYPES):
            obj = self._decode_numpy_types(obj)
        return obj


class Overview:
    FILENAME = "overview.yaml"
    GUTENTAG_LINK = "https://gitlab.hpi.de/akita/guten-tag"

    def __init__(self) -> None:
        self.datasets: List[Dict] = []
        self.seed: Optional[int] = None
        self.git_commit_sha: Optional[str] = None

        if git is not None:
            try:
                self.git_commit_sha = git.Repo(
                    search_parent_directories=True
                ).head.object.hexsha
            except git.InvalidGitRepositoryError:
                pass

    def add_seed(self, seed: Optional[int]) -> None:
        self.seed = seed

    def add_dataset(self, config: Dict) -> None:
        self.datasets.append(config)

    def add_datasets(self, configs: List[Dict]) -> None:
        self.datasets.extend(configs)

    def remove_dataset_by_name(self, name: Union[str, Callable[[str], bool]]) -> None:
        if isinstance(name, str):
            self.datasets = [d for d in self.datasets if d["name"] != name]
        else:
            self.datasets = [d for d in self.datasets if not name(d["name"])]

    def save_to_output_dir(self, path: os.PathLike) -> None:
        overview: Dict[str, Any] = dict()
        overview["generated-timeseries"] = []
        for i, dataset in enumerate(self.datasets):
            dataset["generation-id"] = dataset.get("base_oscillation", {}).get(
                "title", i
            )
            overview["generated-timeseries"].append(dataset)

        overview["meta"] = {}
        overview["meta"]["seed"] = self.seed
        overview["meta"]["git_commit_sha"] = self.git_commit_sha
        overview["meta"]["download_link"] = self.GUTENTAG_LINK

        overview = DictSanitizer().sanitize(overview)

        with open(os.path.join(path, self.FILENAME), "w") as f:
            yaml.dump(overview, f)
