"""V13 benchmark construction with explicit input/oracle boundaries."""
from .config import BenchmarkConfig
from .generation import generate_benchmark

__all__ = ["BenchmarkConfig", "generate_benchmark"]
