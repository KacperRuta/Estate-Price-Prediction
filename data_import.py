from pathlib import Path
import shutil

import pandas as pd


class HousingDataLoader:
    """Download and load the housing dataset from Kaggle."""

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str = "data_cache",
        dataset_file_name: str = "Housing.csv",
    ) -> None:
        self._dataset_name = dataset_name
        self._cache_dir = Path(cache_dir)
        self._dataset_file_name = dataset_file_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self) -> Path:
        """Download the dataset only when it is not available in cache."""
        cached_file = self._cache_dir / self._dataset_file_name
        if cached_file.exists():
            return self._cache_dir

        try:
            import kagglehub
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "kagglehub is required only when the dataset is not available in cache."
            ) from error

        downloaded_path = Path(kagglehub.dataset_download(self._dataset_name))
        csv_files = sorted(downloaded_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in dataset directory: {downloaded_path}")

        matching_csv_file = self._find_dataset_file(csv_files)
        shutil.copy2(matching_csv_file, cached_file)

        return self._cache_dir

    def load_data(self) -> pd.DataFrame:
        """Load the cached dataset CSV file."""
        dataset_path = self.download_dataset()
        csv_file = dataset_path / self._dataset_file_name

        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset file was not found: {csv_file}")

        return pd.read_csv(csv_file)

    def _find_dataset_file(self, csv_files: list[Path]) -> Path:
        for csv_file in csv_files:
            if csv_file.name == self._dataset_file_name:
                return csv_file

        if len(csv_files) == 1:
            return csv_files[0]

        available_files = ", ".join(csv_file.name for csv_file in csv_files)
        raise FileNotFoundError(
            f"Expected '{self._dataset_file_name}', available CSV files: {available_files}"
        )
