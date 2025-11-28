"""Tests for model definitions and code."""

import ast

import pytest

from .posteriordb import (
    REQUIRED_MODEL_INFO_FIELDS,
    PosteriorDB,
    get_models_used_by_posteriors,
    validate_date_format,
)


class TestModelInfoValidity:
    """Test that all model info JSON files are valid."""

    def test_all_models_have_required_fields(self, pdb: PosteriorDB):
        """Each model info must have all required fields."""
        errors = []
        for name in pdb.model_names():
            model_info = pdb.get_model_info(name)
            missing = REQUIRED_MODEL_INFO_FIELDS - set(model_info.keys())
            if missing:
                errors.append(f"{name}: missing fields {missing}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_model_name_matches_filename(self, pdb: PosteriorDB):
        """The 'name' field must match the filename."""
        errors = []
        for name in pdb.model_names():
            model_info = pdb.get_model_info(name)
            if model_info.get("name") != name:
                errors.append(
                    f"{name}: name field '{model_info.get('name')}' != filename '{name}'"
                )

        if errors:
            pytest.fail("\n".join(errors))

    def test_model_date_format(self, pdb: PosteriorDB):
        """The 'added_date' field must be in YYYY-MM-DD format."""
        errors = []
        for name in pdb.model_names():
            model_info = pdb.get_model_info(name)
            date = model_info.get("added_date", "")
            if not validate_date_format(date):
                errors.append(f"{name}: invalid date format '{date}'")

        if errors:
            pytest.fail("\n".join(errors))

    def test_model_has_at_least_one_implementation(self, pdb: PosteriorDB):
        """Each model must have at least one implementation."""
        errors = []
        for name in pdb.model_names():
            model_info = pdb.get_model_info(name)
            implementations = model_info.get("model_implementations", {})
            if not implementations:
                errors.append(f"{name}: no model implementations defined")

        if errors:
            pytest.fail("\n".join(errors))


class TestModelCodeFiles:
    """Test that model code files exist and are accessible."""

    def test_stan_model_code_files_exist(self, pdb: PosteriorDB):
        """Stan model code files referenced in info must exist."""
        errors = []
        for name in pdb.model_names():
            model_info = pdb.get_model_info(name)
            implementations = model_info.get("model_implementations", {})

            if "stan" in implementations:
                code_path = pdb.model_code_path(name, "stan")
                if code_path and not code_path.exists():
                    errors.append(f"{name}: Stan code file not found: {code_path}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_pymc_model_code_files_exist(self, pdb: PosteriorDB):
        """PyMC model code files referenced in info must exist."""
        errors = []
        for name in pdb.model_names():
            model_info = pdb.get_model_info(name)
            implementations = model_info.get("model_implementations", {})

            if "pymc" in implementations:
                code_path = pdb.model_code_path(name, "pymc")
                if code_path and not code_path.exists():
                    errors.append(f"{name}: PyMC code file not found: {code_path}")

        if errors:
            pytest.fail("\n".join(errors))


class TestModelCoverage:
    """Test that all models are used by at least one posterior."""

    def test_all_models_have_posterior(self, pdb: PosteriorDB):
        """Every model must be referenced by at least one posterior."""
        all_models = set(pdb.model_names())
        used_models = get_models_used_by_posteriors(pdb)
        unused_models = all_models - used_models

        if unused_models:
            pytest.fail(
                f"Models not used by any posterior: {sorted(unused_models)}"
            )


class TestStanSyntax:
    """Test Stan model syntax using CmdStanPy."""

    @pytest.fixture(scope="class")
    def cmdstan_available(self):
        """Check if CmdStan is available."""
        try:
            from cmdstanpy import cmdstan_path
            cmdstan_path()
            return True
        except Exception:
            return False

    def test_stan_syntax_valid(self, pdb: PosteriorDB, cmdstan_available):
        """All Stan models must have valid syntax."""
        if not cmdstan_available:
            pytest.skip("CmdStan not available")

        from cmdstanpy import CmdStanModel

        errors = []
        stan_files = pdb.stan_model_files()

        for stan_file in stan_files:
            try:
                # Only check syntax, don't compile
                CmdStanModel(stan_file=str(stan_file), compile=False)
            except Exception as e:
                errors.append(f"{stan_file.name}: {e}")

        if errors:
            pytest.fail("\n".join(errors))


class TestPyMCSyntax:
    """Test PyMC model syntax and structure."""

    def test_pymc_syntax_valid(self, pdb: PosteriorDB):
        """All PyMC models must have valid Python syntax."""
        errors = []
        pymc_files = pdb.pymc_model_files()

        for pymc_file in pymc_files:
            try:
                with open(pymc_file, encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                errors.append(f"{pymc_file.name}: {e}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_pymc_model_function_exists(self, pdb: PosteriorDB):
        """All PyMC models must define a 'model' function that takes 'data'."""
        errors = []
        pymc_files = pdb.pymc_model_files()

        for pymc_file in pymc_files:
            try:
                with open(pymc_file, encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)

                # Find all function definitions
                functions = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]
                function_names = [f.name for f in functions]

                if "model" not in function_names:
                    errors.append(f"{pymc_file.name}: missing 'model' function")
                    continue

                # Check that model function has 'data' parameter
                model_func = next(f for f in functions if f.name == "model")
                arg_names = [arg.arg for arg in model_func.args.args]
                if "data" not in arg_names:
                    errors.append(
                        f"{pymc_file.name}: 'model' function missing 'data' parameter"
                    )
            except SyntaxError:
                # Already caught in syntax test
                pass

        if errors:
            pytest.fail("\n".join(errors))

    @pytest.fixture(scope="class")
    def pymc_available(self):
        """Check if PyMC is available."""
        try:
            import pymc as pm
            return True
        except ImportError:
            return False

    def test_pymc_model_imports(self, pdb: PosteriorDB, pymc_available):
        """All PyMC models must be importable without errors."""
        if not pymc_available:
            pytest.skip("PyMC not available")

        import importlib.util
        import sys

        errors = []
        pymc_files = pdb.pymc_model_files()

        for pymc_file in pymc_files:
            module_name = f"posteriordb_test_{pymc_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, pymc_file
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Check that model function exists and is callable
                if not hasattr(module, "model"):
                    errors.append(f"{pymc_file.name}: no 'model' attribute after import")
                elif not callable(module.model):
                    errors.append(f"{pymc_file.name}: 'model' is not callable")
            except Exception as e:
                errors.append(f"{pymc_file.name}: import error - {e}")
            finally:
                # Clean up
                if module_name in sys.modules:
                    del sys.modules[module_name]

        if errors:
            pytest.fail("\n".join(errors))
