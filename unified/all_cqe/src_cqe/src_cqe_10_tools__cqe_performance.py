"""
CQE Performance Monitor

Tracks system performance metrics including slice validation rates,
memory usage, energy consumption, and operational statistics.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

@dataclass
class OperationMetrics:
    """Metrics for a specific operation type"""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    errors: int = 0

    def add_timing(self, duration: float):
        """Add a timing measurement"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)

    def add_error(self):
        """Record an error"""
        self.errors += 1

    def get_average_time(self) -> float:
        """Get average operation time"""
        return self.total_time / max(self.count, 1)

    def get_recent_average(self) -> float:
        """Get recent average time"""
        if not self.recent_times:
            return 0.0
        return statistics.mean(self.recent_times)

    def get_success_rate(self) -> float:
        """Get operation success rate"""
        total_ops = self.count + self.errors
        return self.count / max(total_ops, 1)

class PerformanceMonitor:
    """System performance monitoring and metrics collection"""

    def __init__(self):
        self.operations: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        self.system_metrics: Dict[str, Any] = {}
        self.start_time = time.time()

        # Slice-specific metrics
        self.slice_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Thread safety
        self.lock = threading.Lock()

        # Auto-collection of system metrics
        self._collect_system_metrics()

    def start_operation(self, operation_type: str) -> str:
        """Start timing an operation"""
        operation_id = f"{operation_type}_{time.time()}_{id(self)}"

        with self.lock:
            self.active_operations[operation_id] = time.time()

        return operation_id

    def end_operation(self, operation_id: str, success: bool = True):
        """End timing an operation"""
        end_time = time.time()

        with self.lock:
            if operation_id in self.active_operations:
                start_time = self.active_operations.pop(operation_id)
                duration = end_time - start_time

                # Extract operation type from ID
                operation_type = operation_id.split('_')[0]

                if success:
                    self.operations[operation_type].add_timing(duration)
                else:
                    self.operations[operation_type].add_error()

    def record_slice_metric(self, slice_name: str, metric_name: str, value: Any):
        """Record a slice-specific metric"""
        with self.lock:
            self.slice_metrics[slice_name][metric_name] = value

    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU and memory
            self.system_metrics.update({
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "uptime_seconds": time.time() - self.start_time
            })

            # Process-specific metrics
            process = psutil.Process()
            self.system_metrics.update({
                "process_memory_mb": process.memory_info().rss / (1024**2),
                "process_cpu_percent": process.cpu_percent(),
                "thread_count": process.num_threads(),
                "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else 0
            })

        except Exception as e:
            print(f"Warning: Could not collect system metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""

        # Update system metrics
        self._collect_system_metrics()

        with self.lock:
            # Operation metrics
            operation_stats = {}
            for op_type, metrics in self.operations.items():
                operation_stats[op_type] = {
                    "count": metrics.count,
                    "total_time": metrics.total_time,
                    "avg_time": metrics.get_average_time(),
                    "recent_avg_time": metrics.get_recent_average(),
                    "min_time": metrics.min_time if metrics.min_time != float('inf') else 0,
                    "max_time": metrics.max_time,
                    "errors": metrics.errors,
                    "success_rate": metrics.get_success_rate()
                }

            # Overall statistics
            total_operations = sum(m.count for m in self.operations.values())
            total_errors = sum(m.errors for m in self.operations.values())

            return {
                "system": dict(self.system_metrics),
                "operations": operation_stats,
                "slices": dict(self.slice_metrics),
                "summary": {
                    "total_operations": total_operations,
                    "total_errors": total_errors,
                    "overall_success_rate": total_operations / max(total_operations + total_errors, 1),
                    "active_operations": len(self.active_operations),
                    "uptime_hours": self.system_metrics.get("uptime_seconds", 0) / 3600
                }
            }

    def get_slice_performance(self, slice_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific slice"""

        with self.lock:
            slice_ops = {
                k: v for k, v in self.operations.items()
                if slice_name.lower() in k.lower()
            }

            return {
                "slice_name": slice_name,
                "operations": {
                    op: {
                        "count": metrics.count,
                        "avg_time": metrics.get_average_time(),
                        "success_rate": metrics.get_success_rate()
                    }
                    for op, metrics in slice_ops.items()
                },
                "custom_metrics": self.slice_metrics.get(slice_name, {}),
                "total_operations": sum(m.count for m in slice_ops.values()),
                "avg_operation_time": statistics.mean([
                    m.get_average_time() for m in slice_ops.values()
                ]) if slice_ops else 0.0
            }

    def reset_metrics(self):
        """Reset all performance metrics"""
        with self.lock:
            self.operations.clear()
            self.slice_metrics.clear()
            self.active_operations.clear()
            self.start_time = time.time()

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""

        metrics = self.get_metrics()
        health_status = {
            "overall": "healthy",
            "issues": [],
            "warnings": []
        }

        # Check system resources
        if metrics["system"]["memory_percent"] > 90:
            health_status["issues"].append("High memory usage (>90%)")
            health_status["overall"] = "critical"
        elif metrics["system"]["memory_percent"] > 75:
            health_status["warnings"].append("Moderate memory usage (>75%)")
            if health_status["overall"] == "healthy":
                health_status["overall"] = "warning"

        if metrics["system"]["cpu_percent"] > 95:
            health_status["issues"].append("High CPU usage (>95%)")
            health_status["overall"] = "critical"

        # Check operation success rates
        for op_type, op_stats in metrics["operations"].items():
            if op_stats["success_rate"] < 0.9:
                health_status["warnings"].append(f"Low success rate for {op_type}: {op_stats['success_rate']:.1%}")
                if health_status["overall"] == "healthy":
                    health_status["overall"] = "warning"

        # Check for stalled operations
        if metrics["summary"]["active_operations"] > 100:
            health_status["warnings"].append(f"High number of active operations: {metrics['summary']['active_operations']}")
            if health_status["overall"] == "healthy":
                health_status["overall"] = "warning"

        return health_status
