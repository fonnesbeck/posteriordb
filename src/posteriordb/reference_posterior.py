"""ReferencePosterior class for accessing reference draws in the posterior database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from posteriordb.utils import load_json, load_json_or_zip

if TYPE_CHECKING:
    from posteriordb.database import PosteriorDatabase


class ReferencePosterior:
    """Reference posterior draws in the posterior database.

    Provides access to gold-standard posterior samples and diagnostics.

    Example:
        >>> pdb = PosteriorDatabase("posterior_database")
        >>> posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        >>> ref = posterior.reference_posterior
        >>> draws = ref.draws()
        >>> print(draws.keys())
        dict_keys(['mu', 'tau', 'theta'])
    """

    def __init__(self, name: str, db: PosteriorDatabase) -> None:
        """Initialize the ReferencePosterior object.

        Args:
            name: Reference posterior name.
            db: PosteriorDatabase instance.
        """
        self._name = name
        self._db = db
        self._info: dict[str, Any] | None = None
        self._draws: dict[str, list[float]] | None = None

    @property
    def name(self) -> str:
        """Reference posterior name."""
        return self._name

    @property
    def information(self) -> dict[str, Any]:
        """Reference posterior metadata.

        Returns:
            Dictionary containing inference settings, diagnostics,
            and quality checks.
        """
        if self._info is None:
            info_path = (
                self._db.reference_posteriors_path / "info" / f"{self._name}.info.json"
            )
            self._info = load_json(info_path)
        return self._info

    def draws(self) -> dict[str, list[float]]:
        """Load and return the reference posterior draws.

        Returns:
            Dictionary mapping parameter names to lists of draws.
            Each parameter has 10,000 draws (the gold standard).

        Raises:
            FileNotFoundError: If the draws file does not exist.
        """
        if self._draws is None:
            draws_path = (
                self._db.reference_posteriors_path / "draws" / f"{self._name}.json"
            )
            self._draws = load_json_or_zip(draws_path)
        return self._draws

    @property
    def inference(self) -> dict[str, Any]:
        """Inference settings used to generate the reference draws."""
        return self.information.get("inference", {})

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Diagnostic information for the reference draws.

        Includes effective sample sizes, R-hat values, divergences, etc.
        """
        return self.information.get("diagnostics", {})

    @property
    def checks_made(self) -> dict[str, bool]:
        """Quality checks performed on the reference draws.

        Returns:
            Dictionary of check names to boolean results.
        """
        return self.information.get("checks_made", {})

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters in the reference posterior."""
        diag = self.diagnostics
        diag_info = diag.get("diagnostic_information", {})
        return diag_info.get("names", [])

    @property
    def ndraws(self) -> int | None:
        """Number of draws in the reference posterior."""
        return self.diagnostics.get("ndraws")

    @property
    def nchains(self) -> int | None:
        """Number of chains used to generate the reference posterior."""
        return self.diagnostics.get("nchains")

    def __repr__(self) -> str:
        return f"ReferencePosterior({self._name!r})"
