"""
VelocityLLM - Lightweight Online Estimators
No ML training pipeline. Everything here is EMA / exponentially-weighted
regression / percentile windows, updated from real completed requests.

  OutputLengthPredictor  -> expected / p50 / p90 output tokens   (Feature 2)
  OnlineCalibrator       -> TTFT & ITL models learned at runtime (Feature 9)
  KVEstimator            -> current vs predicted-future KV usage (Features 1, 2)
"""

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("velocityllm.estimators")


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(math.ceil(q * len(sorted_values))) - 1))
    return sorted_values[idx]


# --------------------------------------------------------------------------
# Exponentially-weighted linear regression  y = a + b*x
# --------------------------------------------------------------------------
class EWLinearModel:
    """
    Online weighted least squares with forgetting. Seeded with prior
    pseudo-observations so predictions are sane before any data arrives, then
    real measurements gradually dominate (self-calibration).
    """

    def __init__(
        self,
        prior_points: List[Tuple[float, float, float]],
        decay: float = 0.97,
        min_slope: float = 0.0,
        floor: float = 1e-4,
    ):
        self.decay = decay
        self.min_slope = min_slope
        self.floor = floor
        self.sw = self.sx = self.sy = self.sxx = self.sxy = 0.0
        for x, y, w in prior_points:
            self._add(x, y, w)
        self.n_obs = 0

    def _add(self, x: float, y: float, w: float) -> None:
        self.sw += w
        self.sx += w * x
        self.sy += w * y
        self.sxx += w * x * x
        self.sxy += w * x * y

    def update(self, x: float, y: float) -> None:
        if y <= 0 or x < 0:
            return
        d = self.decay
        self.sw, self.sx, self.sy, self.sxx, self.sxy = (
            self.sw * d, self.sx * d, self.sy * d, self.sxx * d, self.sxy * d,
        )
        self._add(x, y, 1.0)
        self.n_obs += 1

    def coefficients(self) -> Tuple[float, float]:
        mean_x = self.sx / self.sw
        mean_y = self.sy / self.sw
        var_x = self.sxx / self.sw - mean_x ** 2
        if var_x < 1e-9:
            slope = 0.0
        else:
            slope = (self.sxy / self.sw - mean_x * mean_y) / var_x
        slope = max(self.min_slope, slope)
        intercept = mean_y - slope * mean_x
        return intercept, slope

    def predict(self, x: float) -> float:
        a, b = self.coefficients()
        return max(self.floor, a + b * max(0.0, x))

    def state(self) -> Dict[str, Any]:
        return {"sw": self.sw, "sx": self.sx, "sy": self.sy, "sxx": self.sxx, "sxy": self.sxy, "n_obs": self.n_obs}

    def load(self, st: Dict[str, Any]) -> None:
        self.sw, self.sx, self.sy = float(st["sw"]), float(st["sx"]), float(st["sy"])
        self.sxx, self.sxy, self.n_obs = float(st["sxx"]), float(st["sxy"]), int(st.get("n_obs", 0))


# --------------------------------------------------------------------------
# Feature 2: output-length predictor (EMA + p50/p90)
# --------------------------------------------------------------------------
@dataclass
class OutputPrediction:
    expected: int   # EMA-based point estimate (used for scheduling math)
    p50: int
    p90: int        # conservative estimate (used for memory-risk math)
    samples: int


class OutputLengthPredictor:
    PROMPT_BINS = (128, 512)  # -> "p_short" / "p_mid" / "p_long"

    def __init__(self, alpha: float = 0.15, window: int = 256, prior_fraction: float = 0.6, prior_cap: int = 256):
        self.alpha = alpha
        self.prior_fraction = prior_fraction
        self.prior_cap = prior_cap
        self._ema: Dict[str, float] = {}
        self._windows: Dict[str, Deque[int]] = {}
        self._window = window
        self.samples = 0

    def _bin(self, prompt_tokens: int) -> str:
        if prompt_tokens <= self.PROMPT_BINS[0]:
            return "p_short"
        if prompt_tokens <= self.PROMPT_BINS[1]:
            return "p_mid"
        return "p_long"

    def observe(self, prompt_tokens: int, output_tokens: int) -> None:
        if output_tokens <= 0:
            return
        key = self._bin(prompt_tokens)
        for k in (key, "global"):
            prev = self._ema.get(k)
            self._ema[k] = float(output_tokens) if prev is None else (self.alpha * output_tokens + (1 - self.alpha) * prev)
            self._windows.setdefault(k, deque(maxlen=self._window)).append(int(output_tokens))
        self.samples += 1

    def predict(self, prompt_tokens: int, max_tokens: int) -> OutputPrediction:
        key = self._bin(prompt_tokens)
        win = self._windows.get(key)
        if not win or len(win) < 5:
            key = "global"
            win = self._windows.get("global")
        if not win or len(win) < 5:
            # Cold start: no data yet -> conservative prior.
            prior = int(max(1, min(max_tokens, max(16, self.prior_fraction * max_tokens), self.prior_cap)))
            return OutputPrediction(expected=prior, p50=prior, p90=max_tokens, samples=0)
        ordered = sorted(win)
        ema = self._ema[key]
        return OutputPrediction(
            expected=int(max(1, min(max_tokens, round(ema)))),
            p50=int(max(1, min(max_tokens, percentile(ordered, 0.5)))),
            p90=int(max(1, min(max_tokens, percentile(ordered, 0.9)))),
            samples=len(win),
        )

    def avg_output(self) -> float:
        return self._ema.get("global", 64.0)

    def state(self) -> Dict[str, Any]:
        return {"ema": self._ema, "windows": {k: list(v) for k, v in self._windows.items()}, "samples": self.samples}

    def load(self, st: Dict[str, Any]) -> None:
        self._ema = {k: float(v) for k, v in st.get("ema", {}).items()}
        self._windows = {k: deque((int(x) for x in v), maxlen=self._window) for k, v in st.get("windows", {}).items()}
        self.samples = int(st.get("samples", 0))


# --------------------------------------------------------------------------
# Feature 9: online self-calibration
# --------------------------------------------------------------------------
class OnlineCalibrator:
    """
    Learns, from live measurements:
      * TTFT as a function of prompt tokens        (prefill cost)
      * ITL (per-token decode latency) as a function of concurrency
    and tracks how wrong its own predictions are (MAPE EMA) so the dashboard
    can show calibration status. State is keyed per model/backend/GPU config.
    """

    def __init__(self, calibration_key: str, warmup_samples: int = 20, converged_mape: float = 0.30):
        self.key = calibration_key
        self.warmup_samples = warmup_samples
        self.converged_mape = converged_mape
        self.ttft_model = EWLinearModel([(0.0, 0.04, 2.0), (1024.0, 0.15, 2.0)], decay=0.97)
        self.itl_model = EWLinearModel([(1.0, 0.012, 2.0), (16.0, 0.016, 2.0)], decay=0.97, min_slope=0.0)
        self.samples = 0
        self.exec_mape = 0.5      # EMA of |pred-actual|/actual for execution latency
        self.total_mape = 0.5     # same for total (queue + exec) latency
        self._mape_alpha = 0.1
        self.bucket_stats: Dict[str, Dict[str, float]] = {}
        # Learned multiplicative bias of the queue-wait prediction (actual / predicted).
        # >1 means we under-predicted waits (e.g. later HIGH/short arrivals overtook queued work).
        self.wait_correction = 1.0
        # Recently observed average concurrency during executions ("workload regime").
        # A request admitted while the system looks idle will usually still execute under the
        # load that is arriving, so predictions must not assume the instantaneous concurrency.
        self.ema_conc = 1.0
        self.last_pairs: Deque[Dict[str, float]] = deque(maxlen=50)  # predicted vs actual, for the UI

    # --- predictions
    def ttft(self, prompt_tokens: int) -> float:
        return self.ttft_model.predict(prompt_tokens)

    def itl(self, concurrency: float) -> float:
        return self.itl_model.predict(max(1.0, concurrency))

    def exec_time(self, prompt_tokens: int, output_tokens: int, concurrency: float) -> float:
        return self.ttft(prompt_tokens) + max(0, output_tokens - 1) * self.itl(concurrency)

    def token_rate(self, concurrency: float) -> float:
        c = max(1.0, concurrency)
        return c / self.itl(c)

    # --- learning
    def observe_wait(self, actual_wait_s: float, predicted_raw_wait_s: float) -> None:
        """Learn systematic bias of the raw wait prediction. Only informative when a wait was predicted."""
        if predicted_raw_wait_s < 0.25:
            return
        ratio = min(4.0, max(0.5, actual_wait_s / predicted_raw_wait_s))
        self.wait_correction = min(3.0, max(0.8, 0.12 * ratio + 0.88 * self.wait_correction))

    def observe(
        self,
        *,
        prompt_tokens: int,
        output_tokens: int,
        ttft_s: Optional[float],
        itl_s: Optional[float],
        avg_concurrency: float,
        predicted_exec_s: float,
        actual_exec_s: float,
        predicted_total_s: float,
        actual_total_s: float,
        bucket: str,
    ) -> Dict[str, float]:
        if ttft_s and ttft_s > 0:
            self.ttft_model.update(prompt_tokens, ttft_s)
        if itl_s and itl_s > 0:
            self.itl_model.update(max(1.0, avg_concurrency), itl_s)
        self.samples += 1
        if avg_concurrency > 0:
            self.ema_conc = 0.15 * avg_concurrency + 0.85 * self.ema_conc

        exec_err = abs(predicted_exec_s - actual_exec_s) / max(actual_exec_s, 1e-3)
        total_err = abs(predicted_total_s - actual_total_s) / max(actual_total_s, 1e-3)
        a = self._mape_alpha if self.samples > 1 else 1.0
        self.exec_mape = a * exec_err + (1 - a) * self.exec_mape
        self.total_mape = a * total_err + (1 - a) * self.total_mape

        st = self.bucket_stats.setdefault(bucket, {"n": 0.0, "ema_exec_s": actual_exec_s, "ema_tokens": float(output_tokens)})
        st["n"] += 1
        st["ema_exec_s"] = 0.2 * actual_exec_s + 0.8 * st["ema_exec_s"]
        st["ema_tokens"] = 0.2 * output_tokens + 0.8 * st["ema_tokens"]

        pair = {
            "predicted_exec_ms": round(predicted_exec_s * 1000, 1),
            "actual_exec_ms": round(actual_exec_s * 1000, 1),
            "error_pct": round(exec_err * 100, 1),
        }
        self.last_pairs.append(pair)
        return pair

    @property
    def status(self) -> str:
        if self.samples < self.warmup_samples:
            return "warming_up"
        return "calibrated" if self.exec_mape <= self.converged_mape else "converging"

    def snapshot(self) -> Dict[str, Any]:
        ia, ib = self.itl_model.coefficients()
        ta, tb = self.ttft_model.coefficients()
        return {
            "key": self.key,
            "status": self.status,
            "samples": self.samples,
            "warmup_samples": self.warmup_samples,
            "wait_correction": round(self.wait_correction, 3),
            "regime_concurrency": round(self.ema_conc, 2),
            "exec_mape_pct": round(self.exec_mape * 100, 1),
            "total_mape_pct": round(self.total_mape * 100, 1),
            "itl_ms_at_1": round(self.itl(1) * 1000, 2),
            "itl_ms_at_16": round(self.itl(16) * 1000, 2),
            "itl_slope_ms_per_stream": round(ib * 1000, 4),
            "ttft_ms_at_0_tokens": round(ta * 1000, 2),
            "ttft_ms_per_1k_prompt_tokens": round(tb * 1000 * 1000, 2),
            "buckets": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.bucket_stats.items()},
            "recent_predictions": list(self.last_pairs)[-10:],
        }

    # --- persistence (optional warm start, keyed by model/backend/GPU)
    def state(self) -> Dict[str, Any]:
        return {
            "ttft": self.ttft_model.state(), "itl": self.itl_model.state(),
            "samples": self.samples, "exec_mape": self.exec_mape, "total_mape": self.total_mape,
            "wait_correction": self.wait_correction, "ema_conc": self.ema_conc,
        }

    def load(self, st: Dict[str, Any]) -> None:
        self.ttft_model.load(st["ttft"])
        self.itl_model.load(st["itl"])
        self.samples = int(st.get("samples", 0))
        self.exec_mape = float(st.get("exec_mape", 0.5))
        self.total_mape = float(st.get("total_mape", 0.5))
        self.wait_correction = float(st.get("wait_correction", 1.0))
        self.ema_conc = float(st.get("ema_conc", 1.0))


def save_calibration(path: str, calibrator: OnlineCalibrator, predictor: OutputLengthPredictor) -> None:
    try:
        data: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        data[calibrator.key] = {"calibrator": calibrator.state(), "predictor": predictor.state()}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception as exc:  # never let persistence break serving
        logger.warning("Could not save calibration state: %s", exc)


def load_calibration(path: str, calibrator: OnlineCalibrator, predictor: OutputLengthPredictor) -> bool:
    try:
        if not path or not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)
        entry = data.get(calibrator.key)
        if not entry:
            return False
        calibrator.load(entry["calibrator"])
        predictor.load(entry["predictor"])
        logger.info("Loaded calibration for %s (%d samples).", calibrator.key, calibrator.samples)
        return True
    except Exception as exc:
        logger.warning("Could not load calibration state: %s", exc)
        return False


# --------------------------------------------------------------------------
# Features 1 & 2: token accounting + future KV pressure
# --------------------------------------------------------------------------
@dataclass
class RunningInfo:
    """Live bookkeeping for one executing request."""
    request_id: str
    prompt_tokens: int
    max_tokens: int
    expected_output: int
    p90_output: int
    bucket: str
    traffic_class: str
    start_time: float
    generated: int = 0
    predicted_exec_s: float = 0.0   # prediction made AT DISPATCH (used to score the calibrator)


@dataclass
class KVSnapshot:
    current_tokens: int            # prompt + generated, summed over running requests
    predicted_future_tokens: int   # running requests grown to their predicted length
    queued_tokens: int             # admitted-but-waiting requests (will need KV once dispatched)
    capacity_tokens: int
    safe_tokens: int
    bytes_per_token: int
    source: str = "estimate"

    @property
    def current_pressure(self) -> float:
        return self.current_tokens / max(1, self.capacity_tokens)

    @property
    def future_pressure(self) -> float:
        """Predicted pressure vs the SAFE budget (>=1.0 means we expect to exceed it)."""
        return (self.predicted_future_tokens + self.queued_tokens) / max(1, self.safe_tokens)

    def to_dict(self) -> Dict[str, Any]:
        mb = lambda t: round(t * self.bytes_per_token / (1024 * 1024), 1)
        return {
            "source": self.source,
            "current_tokens": self.current_tokens,
            "predicted_future_tokens": self.predicted_future_tokens,
            "queued_tokens": self.queued_tokens,
            "capacity_tokens": self.capacity_tokens,
            "safe_tokens": self.safe_tokens,
            "current_mb": mb(self.current_tokens),
            "predicted_future_mb": mb(self.predicted_future_tokens + self.queued_tokens),
            "capacity_mb": mb(self.capacity_tokens),
            "current_pressure": round(self.current_pressure, 4),
            "predicted_future_pressure": round(self.future_pressure, 4),
        }


class KVEstimator:
    def __init__(self, server_cfg: Any, smart_cfg: Any, predictor: OutputLengthPredictor):
        self.server_cfg = server_cfg
        self.smart = smart_cfg
        self.predictor = predictor
        self.engine_capacity_tokens: int = 0
        self.source = "estimate"

    def capacity_tokens(self, gpu_total_mb: int) -> int:
        if self.smart.kv_capacity_tokens_override > 0:
            return self.smart.kv_capacity_tokens_override
        if self.engine_capacity_tokens > 0:
            return self.engine_capacity_tokens
        util = getattr(self.server_cfg, "gpu_memory_utilization", 0.80)
        usable_mb = max(0.0, gpu_total_mb * util - self.smart.model_weights_mb)
        tokens = int(usable_mb * 1024 * 1024 / max(1, self.smart.kv_bytes_per_token))
        return max(4096, tokens)

    def snapshot(self, running: Iterable[RunningInfo], queued_tokens: int, gpu_total_mb: int) -> KVSnapshot:
        cap = self.capacity_tokens(gpu_total_mb)
        current = 0
        future = 0
        for r in running:
            current += r.prompt_tokens + r.generated
            pred = self.predictor.predict(r.prompt_tokens, r.max_tokens)
            if r.generated < pred.expected:
                total_out = pred.expected
            else:
                # already longer than expected -> assume it keeps going toward p90 / max_tokens
                total_out = min(r.max_tokens, max(pred.p90, int(r.generated * 1.25)))
            future += r.prompt_tokens + max(r.generated, total_out)
        safe = int(cap * self.smart.kv_safe_fraction)
        return KVSnapshot(current, future, int(queued_tokens), cap, safe, self.smart.kv_bytes_per_token, self.source)
