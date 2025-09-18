"""Progress reporting and timing utilities."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import Callable, Optional

from matplotlib.pyplot import close as pl_close, figure as pl_figure, show as pl_show

from .string_ops import strip_zero


class Progress:
    """Simple command-line progress bar."""

    def __init__(self, N: int = 1, format: str = "%", indent: int = 3, min: int = 0):
        self.start_time = datetime.now()
        self.N = N
        self.n = 0
        self.n0 = 0
        self.min = min
        self.format = self.format_percent
        if "%" in format:
            self.format = self.format_percent
        if "#" in format:
            self.format = self.format_bar
            try:
                n = int(format.split("#")[0])
            except ValueError:
                n = 1
            self.bar_length = n
        self.indent = indent * " "
        self.eta: Optional[timedelta] = None
        self.time_last_eta: Optional[datetime] = None
        self.time_str = "--:--:--"
        self.length = 0
        self.prev_n = -1

    def __str__(self) -> str:
        return ", ".join(f"{k}:{v}" for k, v in self.__dict__.items() if k[0] != "_" and not callable(v))

    def set_min(self, min_value: int):
        self.reset_time()
        self.min = self.n0 = min_value

    def reset(self, N: int = 1, **kwargs):
        self.N = N
        self.n0 = 0
        self.reset_time(**kwargs)

    def reset_time(self, n: int = 0, min: Optional[int] = None):
        self.start_time = datetime.now()
        self.min = min or 0
        self.time_str = "--:--:--"
        self.n0 = max(n, self.min)
        self.prev_n = -1
        self.eta = None
        self.time_last_eta = None

    def format_percent(self, n: int) -> str:
        nn = max(n - self.min, 0)
        percent = 100 * nn / (self.N - self.min)
        return f"Progress {n: 4d} / {self.N:4d} = {percent:.0f} %   ETA: {self.eta}"

    def format_bar(self, n: int) -> str:
        return f"{self.fraction(n)}  [{self.bar(n)}]  {self.time_str}"

    def bar(self, n: int) -> str:
        hash_count = 0
        nn = max(n - self.min, 0)
        if (diff := self.N - self.min) > 0:
            hash_count = int(self.bar_length * nn / diff)
        rest = self.bar_length - hash_count
        if hash_count <= self.bar_length:
            return f"{'#' * hash_count}{'-' * rest}"
        return f"-- E R R O R, n:{n}, N:{self.N}, min:{self.min} --"

    def fraction(self, n: Optional[int]) -> str:
        n = n or 0
        nn = max(n - self.min, 0)
        t, T = strip_zero((n, self.N))
        if self.min > 0 and n >= self.min:
            a, b = strip_zero((self.min, nn))
            t = f"({a} + {b})"
        return f"{t} / {T}"

    def set_N(self, N: int):
        self.N = N

    def print(self, n: int = -1, head: str | None = None, text: str | None = None):
        if n < 0:
            self.n += 1
            n = self.n
        self.remaining_time(n)
        line = self.format(n)
        trail_space = max(1, self.length - len(line))
        self.length = len(line)
        print(
            "\r"
            + (head + " " if head else "")
            + self.indent
            + line
            + (" " + text if text else "")
            + trail_space * " ",
            end="",
            flush=True,
        )

    def remaining_time(self, n: int) -> str:
        time_ = timedelta(0)
        if self.prev_n < self.min:
            self.start_time = datetime.now()
        if n > self.prev_n and n > self.min and n > self.n0:
            nn = n - self.n0
            eta = max(int((self.N - n) * (datetime.now() - self.start_time).total_seconds() / nn), 0)
            self.eta = timedelta(seconds=eta)
            self.time_last_eta = datetime.now()
            time_ = self.eta
        elif self.eta:
            time_ = self.eta - (datetime.now() - self.time_last_eta)
        self.prev_n = n
        time_ = max(timedelta(0), time_)
        self.time_str = str(time_).split(".")[0]
        return self.time_str


class Timer:
    """Simple execution timer writing results to ``filename``."""

    def __init__(self, filename: str | None = None):
        self.counter = 0
        self.timefile = Path(f"{filename}_timer.dat")
        self.timefile.write_text("# step \t seconds\n")
        self.starttime = time()
        self.info = f"Execution time saved in {self.timefile.name}"

    def start(self):
        self.counter += 1
        self.starttime = time()

    def stop(self):
        with self.timefile.open("a") as f:
            f.write(f"{self.counter:d}\t{time() - self.starttime:.3e}\n")


class TimerThread:
    """Background timer thread executing a callback after a timeout."""

    DEBUG = False

    def __init__(self, limit: float = 0, prec: float = 0.5, func: Optional[Callable] = None):
        self._func = func
        self._call_func = func
        self._limit = limit
        self._idle = prec
        self._running = False
        self._starttime: Optional[datetime] = None
        self._endtime: Optional[float] = None
        self._thread = Thread(target=self._timer, daemon=True)
        self.DEBUG and print(f"Creating {self}")

    def __str__(self) -> str:
        func_name = self._func.__qualname__ if self._func else None
        return f"<TimerThread (limit={self._limit}, prec={self._idle}, func={func_name}, thread={self._thread})>"

    def __del__(self):
        self.DEBUG and print(f"Deleting {self}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def endtime(self):
        return self._endtime

    def uptime(self):
        return self._limit - self.time()

    def start(self):
        self._endtime = None
        self._call_func = self._func
        self._starttime = datetime.now()
        if not self._running:
            self._running = True
            self._thread.start()

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join()

    def cancel_if_alive(self):
        if not self._endtime:
            self._call_func = lambda: None
            self._endtime = self.time()
            return True
        return False

    def is_alive(self):
        return not self._endtime

    def time(self):
        if self._starttime is None:
            return 0.0
        return (datetime.now() - self._starttime).total_seconds()

    def _timer(self):
        while self._running:
            sleep(self._idle)
            if not self._endtime:
                elapsed = self.time()
                if elapsed >= self._limit:
                    if self._call_func:
                        self._call_func()
                    self._endtime = elapsed


class LivePlot:
    """Matplotlib based live plot for Jupyter notebooks."""

    def __init__(self, figure: int = 1, func: Optional[Callable] = None, loop=None, **kwargs):
        from IPython import get_ipython

        if ipython := get_ipython():
            ipython.run_line_magic("matplotlib", "widget")
        else:
            raise SystemError("ERROR! LivePlot can only be used inside a Jupyter Notebook/IPython session")
        pl_close("all")
        self.fig = pl_figure(figure)
        canvas = self.fig.canvas
        canvas.header_visible = False
        pl_show()
        self.func = func
        self.kwargs = kwargs
        self.running = False
        self.loop = loop

    def start(self, wait: float = 1.0):
        import asyncio

        async def update():
            self.running = True
            while self.running:
                if self.func:
                    self.func(**self.kwargs)
                self.fig.canvas.draw_idle()
                await asyncio.sleep(wait)

        if self.loop:
            self.loop.run_until_complete(update())
        else:
            loop = asyncio.get_running_loop()
            loop.create_task(update())

    def stop(self):
        if self.running:
            print("Stopping LivePlot!")
            self.running = False
        else:
            print("LivePlot is not running!")


__all__ = ["LivePlot", "Progress", "Timer", "TimerThread"]
