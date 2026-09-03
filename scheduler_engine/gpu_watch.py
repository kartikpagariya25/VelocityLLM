import subprocess
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
    return int(util), int(mem_used), int(mem_total)


print(f"{'TIME':<10}{'GPU UTIL %':<15}{'MEM USED (MB)':<18}{'MEM TOTAL (MB)':<18}")
print("-" * 60)

try:
    while True:
        util, mem_used, mem_total = get_gpu_stats()
        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp:<10}{util:<15}{mem_used:<18}{mem_total:<18}")
        time.sleep(1)
except KeyboardInterrupt:
    pass
