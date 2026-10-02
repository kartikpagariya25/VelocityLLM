import asyncio
import collections
import os
import signal
import socket
import sys

import aiohttp

from .config import ROOT, context_limit


class EngineError(Exception):
    def __init__(self, message, tail=None):
        super().__init__(message)
        self.tail = list(tail or [])


def free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise EngineError(f"No free port found from {start} to {start + 199}")


class Engine:
    def __init__(self, settings, policy, model_path, sla_ms, port, mock, on_line, log_path=None):
        self.settings = settings
        self.policy = policy
        self.model_path = model_path
        self.sla_ms = sla_ms
        self.port = port
        self.mock = mock
        self.on_line = on_line
        self.log_path = log_path
        self.proc = None
        self.reader = None
        self.tail = collections.deque(maxlen=40)
        self.base = f"http://127.0.0.1:{port}"

    def command(self):
        cmd = [
            sys.executable, "-m", "scheduler_engine.server",
            "--policy", self.policy, "--sla-ms", str(self.sla_ms),
            "--port", str(self.port), "--host", "127.0.0.1",
            "--model-path", self.model_path,
            "--max-model-len", str(context_limit(self.model_path)),
        ]
        if self.mock:
            cmd.append("--mock")
        return cmd

    async def _read(self):
        logf = None
        if self.log_path:
            try:
                logf = open(self.log_path, "ab")
            except OSError:
                logf = None
        try:
            while True:
                raw = await self.proc.stdout.readline()
                if not raw:
                    return
                line = raw.decode(errors="replace").rstrip("\r\n")
                if not line.strip():
                    continue
                self.tail.append(line)
                if logf:
                    logf.write(raw)
                    logf.flush()
                await self.on_line(line)
        finally:
            if logf:
                logf.close()

    async def start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.command(), cwd=str(ROOT), env=env, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True, limit=1 << 20,
            )
        except (OSError, ValueError) as e:
            raise EngineError(f"Could not launch the engine process: {e}")
        self.reader = asyncio.create_task(self._read())
        deadline = asyncio.get_running_loop().time() + self.settings.startup_timeout
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                if self.proc.returncode is not None:
                    await asyncio.sleep(0.2)
                    raise EngineError(f"Engine exited during startup (code {self.proc.returncode})", self.tail)
                try:
                    async with session.get(f"{self.base}/health") as r:
                        if r.status == 200:
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    pass
                if asyncio.get_running_loop().time() > deadline:
                    raise EngineError(f"Engine was not healthy after {int(self.settings.startup_timeout)} s", self.tail)
                await asyncio.sleep(1.0)

    async def stats(self):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{self.base}/stats") as r:
                    if r.status == 200:
                        return await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
            return None
        return None

    @property
    def exited(self):
        return self.proc is not None and self.proc.returncode is not None

    async def wait_exit(self):
        await self.proc.wait()
        return self.proc.returncode

    async def stop(self):
        proc = self.proc
        if proc is not None and proc.returncode is None:
            self._signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), 40)
            except asyncio.TimeoutError:
                self._signal(signal.SIGKILL)
                try:
                    await asyncio.wait_for(proc.wait(), 10)
                except asyncio.TimeoutError:
                    pass
        if self.reader:
            try:
                await asyncio.wait_for(self.reader, 5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self.reader.cancel()

    def _signal(self, sig):
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.send_signal(sig)
            except ProcessLookupError:
                pass
