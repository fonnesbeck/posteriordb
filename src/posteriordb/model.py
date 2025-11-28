"""Model class for accessing models in the posterior database."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from posteriordb.utils import load_json

if TYPE_CHECKING:
    from posteriordb.database import PosteriorDatabase


class Model:
    """A statistical model in the posterior database.

    Provides access to model code (Stan, PyMC) and metadata with lazy loading.

    Example:
        >>> pdb = PosteriorDatabase("posterior_database")
        >>> model = pdb.model("eight_schools_noncentered")
        >>> print(model.code("stan"))
        data {
          int<lower=0> J;
          ...
        }
    """

    def __init__(self, name: str, db: PosteriorDatabase) -> None:
        """Initialize the Model object.

        Args:
            name: Model name.
            db: PosteriorDatabase instance.
        """
        self._name = name
        self._db = db
        self._info: dict[str, Any] | None = None
        self._code_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def information(self) -> dict[str, Any]:
        """Model metadata.

        Returns:
            Dictionary containing model metadata including title,
            description, references, and implementation details.
        """
        if self._info is None:
            info_path = self._db.models_info_path / f"{self._name}.info.json"
            self._info = load_json(info_path)
        return self._info

    def code_path(self, framework: str = "stan") -> Path | None:
        """Get the path to the model code file.

        Args:
            framework: Framework name ('stan' or 'pymc').

        Returns:
            Path to the model code file, or None if not available.
        """
        implementations = self.information.get("model_implementations", {})
        impl = implementations.get(framework, {})
        code_path = impl.get("model_code")
        if code_path:
            return self._db.path / code_path
        return None

    def code(self, framework: str = "stan") -> str:
        """Load and return the model code.

        Args:
            framework: Framework name ('stan' or 'pymc').

        Returns:
            Model code as a string.

        Raises:
            ValueError: If the framework is not available for this model.
            FileNotFoundError: If the code file does not exist.
        """
        if framework in self._code_cache:
            return self._code_cache[framework]

        path = self.code_path(framework)
        if path is None:
            available = list(self.information.get("model_implementations", {}).keys())
            raise ValueError(
                f"Framework '{framework}' not available for model '{self._name}'. "
                f"Available: {available}"
            )

        with open(path, encoding="utf-8") as f:
            code = f.read()

        self._code_cache[framework] = code
        return code

    @property
    def frameworks(self) -> list[str]:
        """List of available frameworks for this model."""
        return list(self.information.get("model_implementations", {}).keys())

    @property
    def title(self) -> str | None:
        """Model title, if available."""
        return self.information.get("title")

    @property
    def description(self) -> str | None:
        """Model description, if available."""
        return self.information.get("description")

    @property
    def references(self) -> list[str]:
        """Bibliography references for this model."""
        refs = self.information.get("references", [])
        if isinstance(refs, str):
            return [refs]
        return refs

    @property
    def keywords(self) -> list[str]:
        """Keywords associated with this model."""
        kw = self.information.get("keywords", [])
        if isinstance(kw, str):
            return [kw]
        return kw

    def __repr__(self) -> str:
        return f"Model({self._name!r})"
