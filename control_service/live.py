import secrets
import socket
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .loadgen import build_plan

ONLINE_WINDOW_S = 12.0
MAX_DEVICES = 6
MAX_PER_DEVICE = 100


class DeviceSpec(BaseModel):
    requests: int = Field(default=20, ge=1, le=MAX_PER_DEVICE)
    scenario: Literal["flood", "mixed", "steady", "burst"] = "flood"
    prompt_preset: Literal["short", "medium", "long", "custom"] = "medium"
    prompt_text: Optional[str] = Field(default=None, max_length=500)


class LiveConfig(BaseModel):
    model: Optional[str] = Field(default=None, max_length=120)
    sla_ms: float = Field(default=8000, ge=1000, le=60000)
    repeats: int = Field(default=1, ge=1, le=5)
    policies: list[Literal["static", "dynamic", "smart"]] = Field(default_factory=lambda: ["static", "dynamic"], min_length=1, max_length=3)
    auto_start: bool = True
    window_s: float = Field(default=3.0, ge=0.5, le=20.0)
    seed: int = Field(default=42, ge=0, le=1_000_000)

    @field_validator("policies")
    @classmethod
    def unique_policies(cls, v):
        return list(dict.fromkeys(v))


def choose_model(models, preferred=None):
    ids = [m["id"] for m in models]
    if preferred in ids:
        return preferred
    real = [i for i in ids if not i.startswith("mock")]
    pool = real or ids
    return next((i for i in pool if "3b" in i.lower()), pool[0] if pool else None)


class LiveError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Device:
    def __init__(self, device_id, name):
        self.id = device_id
        self.name = name
        self.spec = None
        self.joined = time.time()
        self.last_seen = time.time()

    def online(self):
        return time.time() - self.last_seen <= ONLINE_WINDOW_S

    def public(self):
        return {"name": self.name, "spec": self.spec, "online": self.online(), "ready": self.spec is not None}


class LiveHub:
    def __init__(self):
        self.devices = {}
        self.last_run_id = None
        self.config = LiveConfig()
        self.timer = None
        self.fire_at = None
        self.last_error = None

    def countdown(self):
        if self.fire_at is None:
            return None
        return max(0.0, round(self.fire_at - time.time(), 1))

    def _unique_name(self, name):
        taken = {d.name.lower() for d in self.devices.values()}
        if name.lower() not in taken:
            return name
        for i in range(2, 100):
            candidate = f"{name} {i}"
            if candidate.lower() not in taken:
                return candidate
        return f"{name} {secrets.token_hex(2)}"

    def join(self, name):
        name = " ".join(name.split())[:24]
        if not name:
            raise ValueError("A device needs a name")
        self._drop_stale()
        if len(self.devices) >= MAX_DEVICES:
            raise OverflowError(f"At most {MAX_DEVICES} devices can join one session")
        device = Device(secrets.token_urlsafe(9), self._unique_name(name))
        self.devices[device.id] = device
        return device

    def _drop_stale(self):
        cutoff = time.time() - 600
        for key in [k for k, d in self.devices.items() if d.last_seen < cutoff]:
            del self.devices[key]

    def get(self, device_id):
        device = self.devices.get(device_id)
        if device:
            device.last_seen = time.time()
        return device

    def leave(self, device_id):
        self.devices.pop(device_id, None)

    def reset(self):
        for device in self.devices.values():
            device.spec = None

    def ready_devices(self):
        return [d for d in self.devices.values() if d.spec and d.online()]

    def total_requests(self, devices=None):
        return sum(d.spec["requests"] for d in (devices if devices is not None else self.ready_devices()))

    def snapshot(self):
        return [d.public() for d in sorted(self.devices.values(), key=lambda d: d.joined)]


def live_plan(devices, seed, vision=False, image_mix="mixed"):
    plan = []
    for i, device in enumerate(devices):
        spec = device["spec"]
        items = build_plan(spec["scenario"], spec["requests"], seed + i * 101, spec["prompt_preset"], spec.get("prompt_text"),
                           vision=vision, image_mix=image_mix)
        for item in items:
            item["device"] = device["name"]
        plan.extend(items)
    plan.sort(key=lambda item: item.get("arrival_time", 0.0))
    return plan


def rows_by_device(rows):
    out = {}
    for row in rows:
        name = row.get("device")
        if name:
            out.setdefault(name, []).append(row)
    return out


def lan_addresses():
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in found:
                found.append(ip)
    except OSError:
        pass
    return [ip for ip in found if not ip.startswith("127.")]
