"""System and environment helpers."""
from __future__ import annotations

from contextlib import contextmanager
from signal import SIGTERM
from subprocess import check_output
from time import sleep
from typing import Callable, Iterable

from psutil import NoSuchProcess, Process, wait_procs


def get_terminal_environment(var: str, file: str = "~/.bashrc") -> str:
    """Return the environment variable ``var`` after sourcing ``file``."""
    env = check_output(["bash", "-c", f"source {file} && env"], text=True)
    match = (value for value in env.split("\n") if value.startswith(var))
    return next(match, "=").split("=")[1]


def call_if_callable(func: Callable | None, *args, **kwargs):
    """Call ``func`` if it is callable."""
    if callable(func):
        return func(*args, **kwargs)
    return False


def running_jupyter() -> bool:
    """Return ``True`` if running inside IPython/Jupyter."""
    from IPython import get_ipython

    return bool(get_ipython())


@contextmanager
def safezip(*gen: Iterable):
    """Zip generators and ensure they are closed when done."""
    try:
        yield zip(*gen)
    finally:
        for iterator in gen:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


def try_except_loop(*args, limit: int = 1, pause: float = 0.05, error=None, raise_error: bool = True,
                    func: Callable | None = None, log: Callable | None = None, **kwargs):
    """Retry ``func`` up to ``limit`` times, handling ``error``."""
    result = None
    for i in range(limit):
        try:
            result = func(*args, **kwargs)
            break
        except error as err:  # type: ignore[misc]
            if log:
                log(i, err)
            sleep(pause)
    if i == limit - 1 and raise_error:
        raise SystemError(
            f"Unable to complete {func.__qualname__} within {limit} tries during {limit * pause} seconds: {error}"
        )
    return result


def kill_process(pid, signal=SIGTERM, children: bool = False, timeout: int = 5, on_terminate: Callable | None = None):
    """Terminate the process ``pid`` optionally including its children."""
    processes: list[Process] = []
    parent = try_except_loop(pid, func=Process, limit=10, pause=0.05, error=NoSuchProcess)
    if parent is None:
        return []
    if children:
        processes.extend(parent.children(recursive=True))
    processes.append(parent)
    for process in processes:
        try:
            process.send_signal(signal)
        except NoSuchProcess:
            pass
    gone, alive = wait_procs(processes, timeout=timeout, callback=on_terminate)
    for process in alive:
        process.kill()
    return gone + alive


def loop_until(func: Callable[..., bool], *args, limit: int | None = None, pause: float | None = None,
               loop_func: Callable | None = None, **kwargs) -> int:
    """Call ``func`` repeatedly until it returns ``True`` or ``limit`` is reached."""
    n = 0
    loop_func = loop_func or (lambda: None)
    while True:
        if func(*args, **kwargs):
            return n
        if pause:
            sleep(pause)
        n += 1
        if limit and n > limit:
            return -1
        loop_func()


def get_python_version():
    """Return :data:`sys.version_info`."""
    from sys import version_info

    return version_info


def print_dict(adict: dict) -> str:
    """Return ``adict`` as a ``key=value`` string."""
    return ", ".join([f"{key}={value}" for key, value in adict.items()])


def print_error(func: Callable):
    """Decorator printing :class:`SystemError` exceptions from ``func``."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemError as error:
            print("\n   " + str(error) + "\n")

    return wrapper


def assert_python_version(major: int | None = None, minor: int | None = None) -> bool:
    """Return ``True`` if the interpreter version is at least ``major.minor``."""
    from sys import stderr, version_info

    sysmajor, sysminor = version_info[0], version_info[1]
    if major is not None and minor is not None:
        if sysmajor < major or (sysmajor == major and sysminor < minor):
            stderr.write(
                "\n[WARNING] This script requires Python {}.{} or higher, you are using Python {}.{}\n".format(
                    major, minor, sysmajor, sysminor
                )
            )
            return False
    return True


__all__ = [
    "assert_python_version",
    "call_if_callable",
    "get_python_version",
    "get_terminal_environment",
    "kill_process",
    "loop_until",
    "print_dict",
    "print_error",
    "running_jupyter",
    "safezip",
    "try_except_loop",
]
