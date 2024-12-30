import time
import functools
from typing import Callable, Optional
from dataclasses import dataclass
import torch
import psutil
import numpy as np


@dataclass
class PerformanceMetrics:
    total_time: float
    tokens_per_second: float
    peak_memory_gb: float
    avg_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tokens: int
    cuda_peak_memory_gb: Optional[float] = None

    def __str__(self) -> str:
        metrics_str = [
            f"Total time: {self.total_time:.2f}s",
            f"Throughput: {self.tokens_per_second:.2f} tokens/s",
            f"Total tokens: {self.total_tokens}",
            f"Peak CPU memory: {self.peak_memory_gb:.2f} GB",
            f"Average latency: {self.avg_latency_ms:.2f} ms",
            f"P90 latency: {self.p90_latency_ms:.2f} ms",
            f"P95 latency: {self.p95_latency_ms:.2f} ms",
            f"P99 latency: {self.p99_latency_ms:.2f} ms",
        ]
        if self.cuda_peak_memory_gb is not None:
            metrics_str.append(f"Peak CUDA memory: {self.cuda_peak_memory_gb:.2f} GB")
        return "\n".join(metrics_str)


class PerformanceMonitor:
    """
    Performance monitoring context manager and decorator.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.start_time = None
        self.latencies = []
        self.token_counts = []
        self.peak_memory = 0
        self.cuda_peak_memory = 0 if torch.cuda.is_available() else None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.peak_memory = psutil.Process().memory_info().rss / (
            1024 * 1024 * 1024
        )  # GB
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.cuda_peak_memory = torch.cuda.max_memory_allocated() / (
                1024 * 1024 * 1024
            )  # GB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

    def record_token_generation(self, num_tokens: int):
        """
        Record latency for generating tokens.
        """
        current_time = time.perf_counter()
        if self.start_time is not None:
            latency = (current_time - self.start_time) * 1000  # Convert to ms
            self.latencies.append(latency)
            self.token_counts.append(num_tokens)
            self.start_time = current_time

        # Update peak memory usage
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
        self.peak_memory = max(self.peak_memory, current_memory)

        if torch.cuda.is_available():
            current_cuda_memory = torch.cuda.max_memory_allocated() / (
                1024 * 1024 * 1024
            )
            self.cuda_peak_memory = max(self.cuda_peak_memory, current_cuda_memory)

    def get_metrics(self) -> PerformanceMetrics:
        """
        Calculate and return performance metrics.
        """
        if not self.latencies:
            return PerformanceMetrics(
                total_time=0,
                tokens_per_second=0,
                peak_memory_gb=self.peak_memory,
                avg_latency_ms=0,
                p90_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                total_tokens=0,
                cuda_peak_memory_gb=self.cuda_peak_memory,
            )

        total_time = sum(self.latencies) / 1000  # Convert back to seconds
        total_tokens = sum(self.token_counts)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        latencies_arr = np.array(self.latencies)
        return PerformanceMetrics(
            total_time=total_time,
            tokens_per_second=tokens_per_second,
            peak_memory_gb=self.peak_memory,
            avg_latency_ms=np.mean(latencies_arr),
            p90_latency_ms=np.percentile(latencies_arr, 90),
            p95_latency_ms=np.percentile(latencies_arr, 95),
            p99_latency_ms=np.percentile(latencies_arr, 99),
            total_tokens=total_tokens,
            cuda_peak_memory_gb=self.cuda_peak_memory,
        )


def measure_performance(name: str = "default"):
    """
    Decorator for measuring performance metrics of text generation.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceMonitor(name) as monitor:
                # For streaming generation
                if "generate_stream" in func.__name__:
                    for chunk, n_tokens, pos in func(*args, **kwargs):
                        monitor.record_token_generation(n_tokens)
                        yield chunk, n_tokens, pos, monitor.get_metrics()
                # For non-streaming generation
                else:
                    result = func(*args, **kwargs)
                    if isinstance(result, tuple):
                        text, n_tokens = result
                        monitor.record_token_generation(n_tokens)
                        return text, n_tokens, monitor.get_metrics()
                    return result

        return wrapper

    return decorator
