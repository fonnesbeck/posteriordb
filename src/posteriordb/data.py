"""Data class for accessing datasets in the posterior database."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from posteriordb.utils import load_json, load_json_or_zip

if TYPE_CHECKING:
    from posteriordb.database import PosteriorDatabase


class Data:
    """A dataset in the posterior database.

    Provides access to dataset values and metadata with lazy loading.

    Example:
        >>> pdb = PosteriorDatabase("posterior_database")
        >>> data = pdb.data("eight_schools")
        >>> print(data.values())
        {'J': 8, 'y': [28, 8, -3, 7, -1, 1, 18, 12], ...}
    """

    def __init__(self, name: str, db: PosteriorDatabase) -> None:
        """Initialize the Data object.

        Args:
            name: Dataset name.
            db: PosteriorDatabase instance.
        """
        self._name = name
        self._db = db
        self._info: dict[str, Any] | None = None
        self._values: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    def information(self) -> dict[str, Any]:
        """Dataset metadata.

        Returns:
            Dictionary containing dataset metadata including title,
            description, references, and other information.
        """
        if self._info is None:
            info_path = self._db.data_info_path / f"{self._name}.info.json"
            self._info = load_json(info_path)
        return self._info

    def values(self) -> dict[str, Any]:
        """Load and return the dataset values.

        Returns:
            Dictionary containing the dataset values.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        if self._values is None:
            data_file = self.information.get("data_file", "")
            if data_file:
                # data_file is relative to database root
                data_path = self._db.path / data_file
            else:
                # Fall back to standard location
                data_path = self._db.data_path / f"{self._name}.json"

            self._values = load_json_or_zip(data_path)
        return self._values

    @property
    def title(self) -> str | None:
        """Dataset title, if available."""
        return self.information.get("title")

    @property
    def description(self) -> str | None:
        """Dataset description, if available."""
        return self.information.get("description")

    @property
    def references(self) -> list[str]:
        """Bibliography references for this dataset."""
        refs = self.information.get("references", [])
        if isinstance(refs, str):
            return [refs]
        return refs

    @property
    def keywords(self) -> list[str]:
        """Keywords associated with this dataset."""
        kw = self.information.get("keywords", [])
        if isinstance(kw, str):
            return [kw]
        return kw

    def __repr__(self) -> str:
        return f"Data({self._name!r})"
