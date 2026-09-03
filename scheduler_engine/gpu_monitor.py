import subprocess
import threading
import time


def get_gpu_stats():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    util, mem_used, mem_total = result.stdout.strip().split(", ")
    return int(util), int(mem_used)


class GPUMonitor:
    def __init__(self, sample_interval=0.3):
        self.sample_interval = sample_interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                util, mem_used = get_gpu_stats()
                self.samples.append({"gpu_util_percent": util, "mem_used_mb": mem_used})
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def summary(self):
        if not self.samples:
            return None
        utils = [s["gpu_util_percent"] for s in self.samples]
        mems = [s["mem_used_mb"] for s in self.samples]
        return {
            "avg_gpu_util_percent": sum(utils) / len(utils),
            "max_gpu_util_percent": max(utils),
            "min_gpu_util_percent": min(utils),
            "avg_mem_used_mb": sum(mems) / len(mems),
            "max_mem_used_mb": max(mems),
            "sample_count": len(self.samples),
        }
