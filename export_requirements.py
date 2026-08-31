"""Export project dependencies from the active Python environment."""

import ast
import platform
import sys
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
ENVIRONMENT_FILE = PROJECT_ROOT / "environment_info.txt"
MODULE_DISTRIBUTION_OVERRIDES = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
}


def imported_top_level_modules(python_file):
    """Return top-level module names imported by one Python source file."""
    tree = ast.parse(python_file.read_text(encoding="utf-8"), filename=str(python_file))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def detect_distributions():
    """Map imported third-party modules to installed distributions."""
    python_files = sorted(PROJECT_ROOT.glob("*.py"))
    local_modules = {path.stem for path in python_files}
    imported_modules = set()
    for python_file in python_files:
        imported_modules.update(imported_top_level_modules(python_file))

    package_mapper = getattr(metadata, "packages_distributions", None)
    module_to_distributions = package_mapper() if package_mapper else {}
    distributions = set()
    for module in imported_modules:
        if module in local_modules:
            continue
        if module in MODULE_DISTRIBUTION_OVERRIDES:
            distributions.add(MODULE_DISTRIBUTION_OVERRIDES[module])
        else:
            distributions.update(module_to_distributions.get(module, []))
    return sorted(distributions, key=str.lower)


def write_requirements(distributions):
    """Write exact installed versions to requirements.txt."""
    lines = []
    for distribution in distributions:
        try:
            lines.append(f"{distribution}=={metadata.version(distribution)}")
        except metadata.PackageNotFoundError:
            print(f"Warning: version not found for {distribution}")
    REQUIREMENTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def collect_environment_information():
    """Collect interpreter, platform, PyTorch, CUDA, and GPU information."""
    lines = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
    ]
    try:
        import torch

        lines.extend(
            [
                f"PyTorch: {torch.__version__}",
                f"PyTorch CUDA build: {torch.version.cuda}",
                f"CUDA available: {torch.cuda.is_available()}",
                f"CUDA device count: {torch.cuda.device_count()}",
            ]
        )
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                lines.append(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")
    except ImportError:
        lines.append("PyTorch: not installed")
    ENVIRONMENT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main():
    distributions = detect_distributions()
    requirements = write_requirements(distributions)
    environment = collect_environment_information()

    print(f"Requirements written to: {REQUIREMENTS_FILE}")
    for requirement in requirements:
        print(f"  {requirement}")
    print(f"Environment information written to: {ENVIRONMENT_FILE}")
    for item in environment:
        print(f"  {item}")


if __name__ == "__main__":
    main()
