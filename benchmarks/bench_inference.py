"""ASV benchmarks for PyMC inference performance.

These benchmarks measure PyMC/PyTensor performance using posteriordb
models as standardized test cases.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from posteriordb import BenchmarkSuite, PosteriorDatabase


class PyMCInferenceBenchmarks:
    """Benchmarks for PyMC inference performance."""

    timeout = 600  # 10 minutes max per benchmark
    params = ["eight_schools-eight_schools_noncentered"]
    param_names = ["posterior"]

    def setup(self, posterior):
        """Set up the benchmark suite."""
        db_path = Path(__file__).parent.parent / "posterior_database"
        self.pdb = PosteriorDatabase(db_path)
        self.suite = BenchmarkSuite(self.pdb)

    def time_inference(self, posterior):
        """Time PyMC inference (including compilation)."""
        result = self.suite.run_inference(
            posterior,
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
        )
        # Store result for tracking benchmarks
        self._last_result = result

    def track_min_ess_per_second(self, posterior):
        """Track minimum ESS per second."""
        result = self.suite.run_inference(
            posterior,
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
        )
        return result.min_ess_per_second

    def track_divergences(self, posterior):
        """Track number of divergent transitions."""
        result = self.suite.run_inference(
            posterior,
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
        )
        return result.divergences

    def track_max_rhat(self, posterior):
        """Track maximum R-hat (convergence diagnostic)."""
        result = self.suite.run_inference(
            posterior,
            draws=500,
            tune=500,
            chains=2,
            random_seed=42,
        )
        return result.max_rhat

    # Set units for track_ benchmarks
    track_min_ess_per_second.unit = "ESS/s"
    track_divergences.unit = "count"
    track_max_rhat.unit = "R-hat"
