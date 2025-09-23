"""File helper class used throughout :mod:`ECLlib`."""
from __future__ import annotations

from contextlib import contextmanager
from mmap import ACCESS_READ, ACCESS_WRITE, mmap
from pathlib import Path
from shutil import copy

from ..config import DEBUG
from ..utils import head_file, last_line, tail_file

__all__ = ["File"]


class File:
    """High-level convenience wrapper around :class:`pathlib.Path`."""

    def __init__(self, filename, suffix=None, role=None, ignore_suffix_case=False, exists=False):
        if isinstance(filename, File):
            filename = filename.path
        self.path = Path(filename).resolve() if filename else None
        if suffix:
            self.path = self.with_suffix(suffix, ignore_suffix_case, exists)
        self.role = role.strip() + " " if role else ""
        self.debug = DEBUG and self.__class__.__name__ == File.__name__
        if self.debug:
            print(f"Creating {repr(self)}")

    def __repr__(self):
        return f"<{self.__class__.__name__}, file={self.path}, role={self.role or None}>"

    def __str__(self):
        return f"{self.role}{self.name}"

    def __del__(self):
        if self.__class__.__name__ == File.__name__ and self.debug:
            print(f"Deleting {repr(self)}")

    def __getattr__(self, item):
        try:
            attr = getattr(self.path or Path(), item)
        except AttributeError as error:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            ) from error
        if self.path:
            return attr
        if callable(attr):
            return lambda: None
        return None

    @contextmanager
    def mmap(self, write=False):
        filemap = None
        mode = "rb"
        access = ACCESS_READ
        if write:
            mode += "+"
            access = ACCESS_WRITE
        try:
            with open(self.path, mode=mode) as file:
                filemap = mmap(file.fileno(), length=0, access=access)
                yield filemap
        finally:
            if filemap:
                filemap.close()

    def resize(self, start=0, end=0):
        if start >= end:
            raise SyntaxError("'start' must be less than 'end'")
        length = end - start
        size = self.size()
        newsize = size - length
        if newsize == 0:
            self.touch()
            return
        with self.mmap(write=True) as infile:
            infile.move(start, end, size - end)
            infile.flush()
            infile.resize(size - length)

    def binarydata(self, pos=None, raise_error=False):
        if self.is_file():
            with open(self.path, "rb") as file:
                if pos:
                    file.seek(pos[0])
                    return file.read(pos[1] - pos[0])
                return file.read()
        if raise_error:
            raise SystemError(f"File {self} does not exist")
        return b""

    def as_text(self, **kwargs):
        return self.binarydata(**kwargs).decode()

    def delete(self, raise_error=False, echo=False):
        if not self.path:
            return
        try:
            self.path.unlink(missing_ok=True)
        except (PermissionError, FileNotFoundError) as error:
            if raise_error:
                raise SystemError(f"Unable to delete {self}: {error}") from error
        if echo:
            msg = f"Deleted {self}"
            if callable(echo):
                echo(msg)
            else:
                print(msg)

    def rename(self, newname, raise_error=False, echo=False):
        if not self.path:
            return
        if isinstance(newname, File):
            newname = newname.path
        try:
            self.path.rename(newname)
        except (PermissionError, FileNotFoundError) as error:
            if raise_error:
                raise SystemError(f"Unable to rename {self}: {error}") from error
        if echo:
            msg = f"Renamed {self.path} --> {newname}"
            if callable(echo):
                echo(msg)
            else:
                print(msg)

    def is_file(self):
        if not self.path:
            return False
        return self.path.is_file()

    def with_name(self, file):
        if not self.path:
            return
        return (self.path.parent / file).resolve()

    def with_tag(self, head: str = "", tail: str = ""):
        if not self.path:
            return
        return self.path.parent / (head + self.path.stem + tail + self.path.suffix)

    def with_suffix(self, suffix, ignore_case=False, exists=False):
        if not self.path:
            return None
        if suffix[0] != ".":
            raise ValueError(f"Invalid suffix '{suffix}'")
        ext = suffix
        if ignore_case:
            ext = ".[" + "][".join(s + s.swapcase() for s in suffix[1:]) + "]"
        path = next(self.glob(ext), None)
        if not exists and path is None:
            path = self.path.with_suffix(suffix)
        return path

    def glob(self, pattern):
        if not self.path:
            return ()
        return self.path.parent.glob(self.path.stem + pattern)

    def exists(self, raise_error=False):
        if self.is_file():
            return True
        if raise_error:
            if self.path.parent.is_dir():
                raise SystemError(f"ERROR {self} is missing in folder {self.path.parent}")
            raise SystemError(
                f"ERROR {self} not found because folder {self.path.parent} is missing"
            )
        return False

    def __stat(self, attr):
        if not self.path:
            return None
        if not self.path.exists():
            return None
        return getattr(self.path.stat(), attr)

    def size(self):
        return self.__stat("st_size")

    def created(self):
        return self.__stat("st_ctime")

    def modified(self):
        return self.__stat("st_mtime")

    def changed(self):
        return self.__stat("st_ctime")

    def accessed(self):
        return self.__stat("st_atime")

    def read_text(self):
        if self.is_file():
            return self.path.read_text(encoding="utf-8")
        return ""

    def write_text(self, text):
        if not self.path:
            raise SystemError("File path is not set")
        self.path.write_text(text, encoding="utf-8")

    def append_text(self, text):
        if not self.path:
            raise SystemError("File path is not set")
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(text)

    def tail(self, **kwargs):
        return next(tail_file(self.path, **kwargs), "")

    def reversed(self, **kwargs):
        return tail_file(self.path, **kwargs)

    def head(self, **kwargs):
        return next(head_file(self.path, **kwargs), "")

    def lines(self):
        if self.is_file():
            with open(self.path, "r", encoding="utf-8") as file:
                while line := file.readline():
                    yield line
        return ()

    def line_matching(self, word):
        return next((line for line in self.lines() if word in line), None)

    def last_line(self):
        return last_line(self.path)

    def backup(self, tag, overwrite=False):
        backup_file = self.path.with_name(f"{self.stem}{tag}{self.suffix}")
        if overwrite or not backup_file.exists():
            copy(self.path, backup_file)
            return backup_file

    def replace_text(self, text=(), pos=()):
        data = self.binarydata().decode()
        size = len(data)
        for txt, (a, b) in sorted(zip(text, pos), key=lambda x: x[1][0]):
            if b - a == 0:
                txt = "\n" + txt
            shift = len(data) - size
            new_data = data[: a + 1 + shift] + txt + data[b + shift :]
            data = new_data
        self.write_text(data)

    def append_bytes(self, data):
        with open(self.path, "ab") as file:
            file.write(data)

    def write_bytes(self, data):
        with open(self.path, "wb") as file:
            file.write(data)
