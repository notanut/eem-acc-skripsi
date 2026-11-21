"""S3.1 API suite: synthetic API with schema drift, task templates, stream sampler, checker (docs/s31_spec.md)."""
from .drift import POOLS
from .oracle import check, max_steps, min_calls, reference_plan
from .streams import SUITE_VERSION, prefix, sample_stream
from .suite import ApiResponse, ApiSuite, Truth
from .world import World

__all__ = ["POOLS", "SUITE_VERSION", "ApiResponse", "ApiSuite", "Truth", "World", "check", "max_steps",
           "min_calls", "prefix", "reference_plan", "sample_stream"]
