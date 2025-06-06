import time
import functools
from collections import defaultdict
from typing import Dict
import threading
import numpy as np
import logging # Import logging

ENABLE_TIMING = True

_recursion_depths = defaultdict(int)
_start_times = defaultdict(float)

# Get a logger for this module
# Use the root simulation logger if available, otherwise get a default logger
try:
    # Attempt to get the simulation logger configured elsewhere
    from logging_utils import _simulation_logger as logger
    if logger is None: # Fallback if not yet configured
        logger = logging.getLogger(__name__)
except ImportError:
    # Fallback if logging_utils cannot be imported (e.g., during testing)
    logger = logging.getLogger(__name__)

# Set a default level if needed, although the root config should handle it
# logger.setLevel(logging.INFO)

class TimingStats:
    """
    Singleton class to store and report function timing statistics for each simulation step.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.current_step_times = defaultdict(list)
                cls._instance.call_counts = defaultdict(int)
        return cls._instance

    def add_timing(self, func_name: str, execution_time: float):
        """
        Add timing measurement for a function in the current step.

        Parameters
        ----------
        func_name : str
        execution_time : float
        """
        self.current_step_times[func_name].append(execution_time)
        self.call_counts[func_name] += 1

    def get_step_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Returns timing statistics for the current step as a dictionary.
        """
        stats_dict = {}
        for func_name, timings in self.current_step_times.items():
            if timings:
                stats_dict[func_name] = {
                    'mean': np.mean(timings),
                    'median': np.median(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'calls': self.call_counts[func_name],
                    'total_time': sum(timings)
                }
        return stats_dict

    def print_step_report(self):
        """
        Log a formatted report of current step timing statistics using the logger.
        """
        stats = self.get_step_stats()
        if not stats:
            logger.info("No timing statistics recorded for this step.")
            return

        sorted_funcs = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)

        # Log header using logger.info
        header = f"{'Function':<40} {'Total Time (s)':<15} {'Calls':<8} {'Mean (s)':<12} {'Std Dev (s)':<12}"
        logger.info("--- Timing Report --- ")
        logger.info(header)
        logger.info("-" * len(header))

        # Log each function's stats using logger.info
        for func_name, func_stats in sorted_funcs:
            log_line = (f"{func_name:<40} {func_stats['total_time']:<15.4f} {func_stats['calls']:<8} "
                        f"{func_stats['mean']:<12.4f} {func_stats['std']:<12.4f}")
            logger.info(log_line)
        logger.info("--- End Timing Report ---")

    def reset_step(self):
        """
        Reset timing statistics for a new simulation step.
        """
        self.current_step_times.clear()
        self.call_counts.clear()


def time_it(func):
    """
    Decorator to measure function execution time if ENABLE_TIMING is True.
    """
    if not ENABLE_TIMING:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = func.__qualname__
        _recursion_depths[key] += 1
        if _recursion_depths[key] == 1:
            _start_times[key] = time.time()
        try:
            result = func(*args, **kwargs)
            if _recursion_depths[key] == 1:
                execution_time = time.time() - _start_times[key]
                TimingStats().add_timing(key, execution_time)
            return result
        finally:
            _recursion_depths[key] -= 1

    return wrapper
