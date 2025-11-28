"""Benchmarking utilities for posteriordb.

This module provides tools for benchmarking PyMC inference performance
using posteriordb models as standardized test cases. It allows developers
to monitor performance changes as PyMC and PyTensor evolve.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pymc as pm

    from .database import PosteriorDatabase


@dataclass
class HardwareInfo:
    """Hardware and system information for benchmark context."""

    platform: str
    platform_version: str
    architecture: str
    processor: str
    python_version: str
    cpu_count: int | None = None

    @classmethod
    def collect(cls) -> HardwareInfo:
        """Collect current system hardware information."""
        import os

        return cls(
            platform=platform.system(),
            platform_version=platform.release(),
            architecture=platform.machine(),
            processor=platform.processor() or "unknown",
            python_version=platform.python_version(),
            cpu_count=os.cpu_count(),
        )

    def __repr__(self) -> str:
        return f"HardwareInfo({self.platform}, {self.architecture}, Python {self.python_version})"


@dataclass
class InferenceResult:
    """Results from a single inference benchmark run.

    Attributes
    ----------
    posterior_name : str
        Name of the posterior that was sampled.
    sampling_time : float
        Total wall-clock time for sampling in seconds.
    compile_time : float
        Time spent compiling the model in seconds.
    n_draws : int
        Number of draws per chain.
    n_chains : int
        Number of chains.
    n_tune : int
        Number of tuning samples.
    ess : dict[str, float]
        Effective sample size for each parameter.
    rhat : dict[str, float]
        R-hat convergence diagnostic for each parameter.
    divergences : int
        Number of divergent transitions.
    metadata : dict[str, Any]
        Additional metadata (PyMC version, PyTensor version, etc.).
    hardware : HardwareInfo | None
        Hardware information for the benchmark run.
    timestamp : str | None
        ISO format timestamp when the benchmark was run.
    run_id : str | None
        Unique identifier for this benchmark run.
    """

    posterior_name: str
    sampling_time: float
    compile_time: float
    n_draws: int
    n_chains: int
    n_tune: int
    ess: dict[str, float] = field(default_factory=dict)
    rhat: dict[str, float] = field(default_factory=dict)
    divergences: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    hardware: HardwareInfo | None = None
    timestamp: str | None = None
    run_id: str | None = None

    @property
    def total_draws(self) -> int:
        """Total number of posterior draws across all chains."""
        return self.n_draws * self.n_chains

    @property
    def ess_per_second(self) -> dict[str, float]:
        """Effective samples per second for each parameter."""
        if self.sampling_time == 0:
            return {}
        return {k: v / self.sampling_time for k, v in self.ess.items()}

    @property
    def min_ess(self) -> float:
        """Minimum ESS across all parameters."""
        if not self.ess:
            return 0.0
        return min(self.ess.values())

    @property
    def min_ess_per_second(self) -> float:
        """Minimum ESS/second across all parameters."""
        if self.sampling_time == 0 or not self.ess:
            return 0.0
        return self.min_ess / self.sampling_time

    @property
    def max_rhat(self) -> float:
        """Maximum R-hat across all parameters."""
        if not self.rhat:
            return 0.0
        return max(self.rhat.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceResult:
        """Create from dictionary."""
        if data.get("hardware") and isinstance(data["hardware"], dict):
            data["hardware"] = HardwareInfo(**data["hardware"])
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"InferenceResult({self.posterior_name!r}, "
            f"time={self.sampling_time:.2f}s, "
            f"min_ess={self.min_ess:.0f}, "
            f"ess/s={self.min_ess_per_second:.1f})"
        )


@dataclass
class ParameterComparison:
    """Comparison metrics for a single parameter."""

    name: str
    ks_statistic: float
    ks_pvalue: float
    mean_diff: float
    std_diff: float
    passed: bool

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ParameterComparison({self.name!r}, "
            f"KS={self.ks_statistic:.4f}, p={self.ks_pvalue:.4f}, {status})"
        )


@dataclass
class ComparisonResult:
    """Result of comparing draws to reference posterior."""

    posterior_name: str
    parameters: dict[str, ParameterComparison]
    passed: bool
    summary: str

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"ComparisonResult({self.posterior_name!r}, {status}, {self.summary})"


class BenchmarkSuite:
    """Benchmark PyMC inference performance using posteriordb models.

    This class provides tools for:
    - Running PyMC inference on posteriordb models
    - Measuring execution time, ESS/second, and convergence diagnostics
    - Comparing inference results to reference posteriors for correctness

    Use this to monitor PyMC/PyTensor performance across versions.

    Example
    -------
    >>> from posteriordb import PosteriorDatabase, BenchmarkSuite
    >>> pdb = PosteriorDatabase("posterior_database")
    >>> suite = BenchmarkSuite(pdb)
    >>>
    >>> # Run inference benchmark
    >>> result = suite.run_inference("eight_schools-eight_schools_noncentered")
    >>> print(f"Sampling time: {result.sampling_time:.2f}s")
    >>> print(f"Min ESS/second: {result.min_ess_per_second:.1f}")
    >>>
    >>> # Compare to reference posterior
    >>> comparison = suite.compare_to_reference(
    ...     "eight_schools-eight_schools_noncentered",
    ...     idata
    ... )
    >>> print(comparison.summary)
    """

    # Default posteriors for benchmarking
    DEFAULT_POSTERIORS = [
        "eight_schools-eight_schools_noncentered",
    ]

    def __init__(
        self,
        pdb: PosteriorDatabase,
        posteriors: list[str] | None = None,
    ) -> None:
        """Initialize the benchmark suite.

        Parameters
        ----------
        pdb : PosteriorDatabase
            The posterior database instance.
        posteriors : list[str] | None, optional
            List of posterior names to benchmark. Defaults to DEFAULT_POSTERIORS.
        """
        self.pdb = pdb
        self.posteriors = (
            posteriors if posteriors is not None else self.DEFAULT_POSTERIORS
        )

    def run_inference(
        self,
        posterior_name: str,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        **sampler_kwargs: Any,
    ) -> InferenceResult:
        """Run PyMC inference on a posterior and collect performance metrics.

        Parameters
        ----------
        posterior_name : str
            Name of the posterior to sample.
        draws : int, optional
            Number of draws per chain. Default 1000.
        tune : int, optional
            Number of tuning samples. Default 1000.
        chains : int, optional
            Number of chains. Default 4.
        random_seed : int | None, optional
            Random seed for reproducibility.
        **sampler_kwargs
            Additional arguments passed to pm.sample().

        Returns
        -------
        InferenceResult
            Benchmark results including timing and diagnostics.
        """
        import arviz as az
        import pymc as pm

        posterior = self.pdb.posterior(posterior_name)
        data = posterior.data.values()

        # Build the model
        model = self._build_pymc_model(posterior_name, data)

        # Time compilation (first sample call compiles)
        compile_start = time.perf_counter()
        with model:
            # Do a minimal run to trigger compilation
            pm.sample(
                draws=2,
                tune=2,
                chains=1,
                progressbar=False,
                compute_convergence_checks=False,
                random_seed=random_seed,
            )
        compile_time = time.perf_counter() - compile_start

        # Time the actual sampling
        sample_start = time.perf_counter()
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                progressbar=False,
                random_seed=random_seed,
                **sampler_kwargs,
            )
        sampling_time = time.perf_counter() - sample_start

        # Extract diagnostics
        ess = self._compute_ess(idata)
        rhat = self._compute_rhat(idata)
        divergences = self._count_divergences(idata)

        # Collect metadata
        metadata = self._get_version_info()
        metadata["sampler_kwargs"] = sampler_kwargs

        return InferenceResult(
            posterior_name=posterior_name,
            sampling_time=sampling_time,
            compile_time=compile_time,
            n_draws=draws,
            n_chains=chains,
            n_tune=tune,
            ess=ess,
            rhat=rhat,
            divergences=divergences,
            metadata=metadata,
        )

    def run_all(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        **sampler_kwargs: Any,
    ) -> dict[str, InferenceResult]:
        """Run inference benchmarks on all configured posteriors.

        Parameters
        ----------
        draws : int, optional
            Number of draws per chain. Default 1000.
        tune : int, optional
            Number of tuning samples. Default 1000.
        chains : int, optional
            Number of chains. Default 4.
        random_seed : int | None, optional
            Random seed for reproducibility.
        **sampler_kwargs
            Additional arguments passed to pm.sample().

        Returns
        -------
        dict[str, InferenceResult]
            Dictionary mapping posterior names to their results.
        """
        results = {}
        for name in self.posteriors:
            results[name] = self.run_inference(
                name,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                **sampler_kwargs,
            )
        return results

    def compare_to_reference(
        self,
        posterior_name: str,
        idata_or_draws: Any,
        alpha: float = 0.01,
    ) -> ComparisonResult:
        """Compare inference results to reference posterior.

        Uses the Kolmogorov-Smirnov test for each parameter to assess
        whether samples come from the same distribution as the reference.

        Parameters
        ----------
        posterior_name : str
            Name of the posterior.
        idata_or_draws : InferenceData | dict[str, np.ndarray] | list[dict]
            Either an ArviZ InferenceData object, a dictionary mapping
            parameter names to sample arrays, or a list of chain dicts.
        alpha : float, optional
            Significance level for KS test. Default 0.01.

        Returns
        -------
        ComparisonResult
            Comparison result with per-parameter statistics.

        Raises
        ------
        ValueError
            If no reference draws exist for the posterior.
        """
        from scipy import stats

        posterior = self.pdb.posterior(posterior_name)
        ref_draws_raw = posterior.reference_draws()

        if ref_draws_raw is None:
            raise ValueError(f"No reference draws available for {posterior_name}")

        # Convert reference draws from list of dicts to dict of arrays
        ref_draws = self._combine_chain_draws(ref_draws_raw)

        # Convert user draws to dict format
        user_draws = self._extract_draws(idata_or_draws)

        comparisons: dict[str, ParameterComparison] = {}
        all_passed = True

        for param_name, user_samples in user_draws.items():
            # Try to match parameter names (handle indexing differences)
            ref_param = self._find_matching_param(param_name, ref_draws)
            if ref_param is None:
                continue

            user_arr = np.asarray(user_samples).flatten()
            ref_arr = np.asarray(ref_draws[ref_param]).flatten()

            ks_stat, ks_pval = stats.ks_2samp(user_arr, ref_arr)
            mean_diff = float(abs(np.mean(user_arr) - np.mean(ref_arr)))
            std_diff = float(abs(np.std(user_arr) - np.std(ref_arr)))
            passed = ks_pval > alpha

            if not passed:
                all_passed = False

            comparisons[param_name] = ParameterComparison(
                name=param_name,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pval),
                mean_diff=mean_diff,
                std_diff=std_diff,
                passed=passed,
            )

        n_passed = sum(1 for c in comparisons.values() if c.passed)
        summary = (
            f"{n_passed}/{len(comparisons)} parameters passed KS test (alpha={alpha})"
        )

        return ComparisonResult(
            posterior_name=posterior_name,
            parameters=comparisons,
            passed=all_passed,
            summary=summary,
        )

    def _build_pymc_model(self, posterior_name: str, data: dict) -> pm.Model:
        """Build a PyMC model from posteriordb model code.

        Parameters
        ----------
        posterior_name : str
            Name of the posterior.
        data : dict
            Data dictionary for the model.

        Returns
        -------
        pm.Model
            The PyMC model.
        """
        posterior = self.pdb.posterior(posterior_name)
        model_code = posterior.model.code("pymc")

        # Execute the model code to get the model function
        namespace: dict[str, Any] = {}
        exec(model_code, namespace)

        # The model code should define a 'model' function
        if "model" not in namespace:
            raise ValueError(
                f"PyMC model code for {posterior_name} must define a 'model' function"
            )

        return namespace["model"](data)

    def _extract_draws(self, idata_or_draws: Any) -> dict[str, np.ndarray]:
        """Extract draws from various formats to dict of arrays.

        Parameters
        ----------
        idata_or_draws : InferenceData | dict | list
            Input draws in various formats.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to sample arrays.
        """
        # Check if it's an ArviZ InferenceData
        if hasattr(idata_or_draws, "posterior"):
            draws = {}
            posterior = idata_or_draws.posterior
            for var in posterior.data_vars:
                draws[var] = posterior[var].values.flatten()
            return draws

        # Check if it's a list of chain dicts
        if isinstance(idata_or_draws, list):
            return self._combine_chain_draws(idata_or_draws)

        # Assume it's already a dict
        return {k: np.asarray(v) for k, v in idata_or_draws.items()}

    @staticmethod
    def _combine_chain_draws(
        chain_draws: list[dict[str, list]],
    ) -> dict[str, np.ndarray]:
        """Combine draws from multiple chains into single arrays."""
        if not chain_draws:
            return {}

        param_names = list(chain_draws[0].keys())
        combined: dict[str, np.ndarray] = {}

        for param in param_names:
            all_samples = []
            for chain in chain_draws:
                if param in chain:
                    all_samples.extend(chain[param])
            combined[param] = np.array(all_samples)

        return combined

    @staticmethod
    def _find_matching_param(
        param_name: str, ref_draws: dict[str, np.ndarray]
    ) -> str | None:
        """Find matching parameter name in reference draws.

        Handles differences in indexing notation (e.g., theta[0] vs theta[1]).
        """
        if param_name in ref_draws:
            return param_name

        # Try common variations
        # PyMC uses 0-indexed, Stan uses 1-indexed
        import re

        match = re.match(r"(.+)\[(\d+)\]", param_name)
        if match:
            base, idx = match.groups()
            stan_name = f"{base}[{int(idx) + 1}]"
            if stan_name in ref_draws:
                return stan_name

        return None

    def _compute_ess(self, idata: Any) -> dict[str, float]:
        """Compute effective sample size for all parameters."""
        import arviz as az

        ess_data = az.ess(idata)
        result = {}
        for k, v in ess_data.data_vars.items():
            vals = np.asarray(v.values).flatten()
            # For array parameters, take the minimum ESS
            result[str(k)] = float(np.min(vals))
        return result

    def _compute_rhat(self, idata: Any) -> dict[str, float]:
        """Compute R-hat for all parameters."""
        import arviz as az

        rhat_data = az.rhat(idata)
        result = {}
        for k, v in rhat_data.data_vars.items():
            vals = np.asarray(v.values).flatten()
            # For array parameters, take the maximum R-hat
            result[str(k)] = float(np.max(vals))
        return result

    def _count_divergences(self, idata: Any) -> int:
        """Count divergent transitions."""
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            return int(idata.sample_stats.diverging.sum().values)
        return 0

    @staticmethod
    def _get_version_info() -> dict[str, str]:
        """Get version information for PyMC and PyTensor."""
        versions = {}
        try:
            import pymc

            versions["pymc"] = pymc.__version__
        except (ImportError, AttributeError):
            pass
        try:
            import pytensor

            versions["pytensor"] = pytensor.__version__
        except (ImportError, AttributeError):
            pass
        try:
            import numpy

            versions["numpy"] = numpy.__version__
        except (ImportError, AttributeError):
            pass
        return versions


class BenchmarkStore:
    """Store and retrieve benchmark results in JSON format.

    Results are stored as individual JSON files, one per benchmark run,
    with filenames based on a unique run ID. This allows tracking
    performance across different PyMC/PyTensor versions and hardware.

    Example
    -------
    >>> from posteriordb import PosteriorDatabase, BenchmarkSuite, BenchmarkStore
    >>> pdb = PosteriorDatabase("posterior_database")
    >>> suite = BenchmarkSuite(pdb)
    >>> store = BenchmarkStore("benchmarks/results")
    >>>
    >>> # Run and store a benchmark
    >>> result = suite.run_inference("eight_schools-eight_schools_noncentered")
    >>> store.save(result)
    >>>
    >>> # Query stored results
    >>> results = store.query(posterior_name="eight_schools-eight_schools_noncentered")
    >>> for r in results:
    ...     print(f"{r.metadata['pymc']}: {r.min_ess_per_second:.1f} ESS/s")
    >>>
    >>> # Compare performance across PyMC versions
    >>> comparison = store.compare_versions("eight_schools-eight_schools_noncentered")
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the benchmark store.

        Parameters
        ----------
        path : str | Path
            Directory path where benchmark results will be stored.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, result: InferenceResult) -> Path:
        """Save a benchmark result to the store.

        Parameters
        ----------
        result : InferenceResult
            The benchmark result to save.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        # Generate run_id if not present
        if result.run_id is None:
            result.run_id = self._generate_run_id(result)

        # Set timestamp if not present
        if result.timestamp is None:
            result.timestamp = datetime.now(timezone.utc).isoformat()

        # Collect hardware info if not present
        if result.hardware is None:
            result.hardware = HardwareInfo.collect()

        # Save to JSON file
        filename = f"{result.run_id}.json"
        filepath = self.path / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return filepath

    def load(self, run_id: str) -> InferenceResult:
        """Load a benchmark result by run ID.

        Parameters
        ----------
        run_id : str
            The unique run identifier.

        Returns
        -------
        InferenceResult
            The loaded benchmark result.

        Raises
        ------
        FileNotFoundError
            If no result with the given run_id exists.
        """
        filepath = self.path / f"{run_id}.json"
        with open(filepath) as f:
            data = json.load(f)
        return InferenceResult.from_dict(data)

    def list_results(self) -> list[str]:
        """List all stored run IDs.

        Returns
        -------
        list[str]
            List of run IDs.
        """
        return [p.stem for p in self.path.glob("*.json")]

    def query(
        self,
        posterior_name: str | None = None,
        pymc_version: str | None = None,
        pytensor_version: str | None = None,
        platform: str | None = None,
    ) -> list[InferenceResult]:
        """Query stored results with optional filters.

        Parameters
        ----------
        posterior_name : str | None
            Filter by posterior name.
        pymc_version : str | None
            Filter by PyMC version (exact match or prefix).
        pytensor_version : str | None
            Filter by PyTensor version (exact match or prefix).
        platform : str | None
            Filter by platform (e.g., "Linux", "Darwin", "Windows").

        Returns
        -------
        list[InferenceResult]
            List of matching results, sorted by timestamp (newest first).
        """
        results = []

        for run_id in self.list_results():
            try:
                result = self.load(run_id)
            except (json.JSONDecodeError, KeyError):
                continue

            # Apply filters
            if posterior_name and result.posterior_name != posterior_name:
                continue

            if pymc_version:
                result_pymc = result.metadata.get("pymc", "")
                if not result_pymc.startswith(pymc_version):
                    continue

            if pytensor_version:
                result_pytensor = result.metadata.get("pytensor", "")
                if not result_pytensor.startswith(pytensor_version):
                    continue

            if platform and result.hardware:
                if result.hardware.platform != platform:
                    continue

            results.append(result)

        # Sort by timestamp (newest first)
        results.sort(key=lambda r: r.timestamp or "", reverse=True)
        return results

    def compare_versions(
        self,
        posterior_name: str,
        metric: str = "min_ess_per_second",
    ) -> list[dict[str, Any]]:
        """Compare benchmark results across different package versions.

        Parameters
        ----------
        posterior_name : str
            Name of the posterior to compare.
        metric : str
            Metric to compare. Options: "min_ess_per_second", "sampling_time",
            "compile_time", "divergences", "max_rhat".

        Returns
        -------
        list[dict[str, Any]]
            List of comparison entries with version info and metric values,
            sorted by timestamp.
        """
        results = self.query(posterior_name=posterior_name)

        comparisons = []
        for result in results:
            entry = {
                "run_id": result.run_id,
                "timestamp": result.timestamp,
                "pymc": result.metadata.get("pymc"),
                "pytensor": result.metadata.get("pytensor"),
                "numpy": result.metadata.get("numpy"),
                "platform": result.hardware.platform if result.hardware else None,
                "metric": metric,
                "value": getattr(result, metric, None),
                "n_draws": result.n_draws,
                "n_chains": result.n_chains,
            }
            comparisons.append(entry)

        return comparisons

    def get_latest(
        self,
        posterior_name: str,
        pymc_version: str | None = None,
    ) -> InferenceResult | None:
        """Get the most recent benchmark result for a posterior.

        Parameters
        ----------
        posterior_name : str
            Name of the posterior.
        pymc_version : str | None
            Optional PyMC version filter.

        Returns
        -------
        InferenceResult | None
            Most recent result, or None if no results match.
        """
        results = self.query(
            posterior_name=posterior_name,
            pymc_version=pymc_version,
        )
        return results[0] if results else None

    def _generate_run_id(self, result: InferenceResult) -> str:
        """Generate a unique run ID based on result content."""
        # Create a hash from key identifying information
        id_data = {
            "posterior": result.posterior_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pymc": result.metadata.get("pymc", ""),
            "pytensor": result.metadata.get("pytensor", ""),
        }
        id_str = json.dumps(id_data, sort_keys=True)
        hash_hex = hashlib.sha256(id_str.encode()).hexdigest()[:12]

        # Create readable prefix
        pymc_ver = result.metadata.get("pymc", "unknown").replace(".", "_")
        return f"{result.posterior_name}_{pymc_ver}_{hash_hex}"
