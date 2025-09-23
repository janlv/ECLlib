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
        """Initialize the File.

        Args:
            filename: Path-like object or :class:`File` to wrap.
            suffix: Optional suffix enforced on the resulting path.
            role: Description prepended to the printable representation.
            ignore_suffix_case: Whether suffix matching should be case insensitive.
            exists: Whether to reuse an existing path with the desired suffix.
        """
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
        """Return a developer-friendly representation."""
        return f"<{self.__class__.__name__}, file={self.path}, role={self.role or None}>"

    def __str__(self):
        """Return a human-readable representation."""
        return f"{self.role}{self.name}"

    def __del__(self):
        """Handle object cleanup."""
        if self.__class__.__name__ == File.__name__ and self.debug:
            print(f"Deleting {repr(self)}")

    def __getattr__(self, item):
        """Delegate attribute lookups.

        Args:
            item: Name of the missing attribute to proxy to ``path``.
        """
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
        """Memory-map the file.

        Args:
            write: Whether to open the mapping with write access.
        """
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
        """Resize the file to remove a byte range.

        Args:
            start: Starting byte offset of the slice to drop.
            end: Ending byte offset of the slice to drop.
        """
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
        """Return the file contents as bytes.

        Args:
            pos: Optional ``(start, end)`` byte range to read.
            raise_error: Whether to raise if the path does not exist.
        """
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
        """Return the file contents decoded as text.

        Args:
            **kwargs: Keyword arguments forwarded to :meth:`binarydata`.
        """
        return self.binarydata(**kwargs).decode()

    def delete(self, raise_error=False, echo=False):
        """Delete the file from disk.

        Args:
            raise_error: Whether to raise on deletion failures.
            echo: ``False`` to stay silent, truthy value or callable for logging.
        """
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
        """Rename the file on disk.

        Args:
            newname: Destination path or :class:`File` to rename to.
            raise_error: Whether to raise on rename failures.
            echo: ``False`` to stay silent, truthy value or callable for logging.
        """
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
        """Return whether the path points to a file."""
        if not self.path:
            return False
        return self.path.is_file()

    def with_name(self, file):
        """Return the path with a new name.

        Args:
            file: Filename replacing the current ``Path.name``.
        """
        if not self.path:
            return
        return (self.path.parent / file).resolve()

    def with_tag(self, head: str = "", tail: str = ""):
        """Return the path tagged with a prefix and suffix.

        Args:
            head: Text inserted before the stem.
            tail: Text appended between the stem and suffix.
        """
        if not self.path:
            return
        return self.path.parent / (head + self.path.stem + tail + self.path.suffix)

    def with_suffix(self, suffix, ignore_case=False, exists=False):
        """Return the path with the provided suffix.

        Args:
            suffix: Suffix (including dot) to apply.
            ignore_case: Whether to treat suffix lookups as case-insensitive.
            exists: Whether to reuse an existing match instead of constructing one.
        """
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
        """Return filesystem entries matching the pattern.

        Args:
            pattern: Glob pattern relative to the wrapped stem.
        """
        if not self.path:
            return ()
        return self.path.parent.glob(self.path.stem + pattern)

    def exists(self, raise_error=False):
        """Return whether the path exists.

        Args:
            raise_error: Whether to raise when the entry is missing.
        """
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
        """Return the requested stat attribute.

        Args:
            attr: Name of the ``os.stat_result`` attribute to fetch.
        """
        if not self.path:
            return None
        if not self.path.exists():
            return None
        return getattr(self.path.stat(), attr)

    def size(self):
        """Return the file size in bytes."""
        return self.__stat("st_size")

    def created(self):
        """Return the file creation timestamp."""
        return self.__stat("st_ctime")

    def modified(self):
        """Return the file modification timestamp."""
        return self.__stat("st_mtime")

    def changed(self):
        """Return the file change timestamp."""
        return self.__stat("st_ctime")

    def accessed(self):
        """Return the file access timestamp."""
        return self.__stat("st_atime")

    def read_text(self):
        """Read the file as text."""
        if self.is_file():
            return self.path.read_text(encoding="utf-8")
        return ""

    def write_text(self, text):
        """Write text to disk."""
        if not self.path:
            raise SystemError("File path is not set")
        self.path.write_text(text, encoding="utf-8")

    def append_text(self, text):
        """Append text to the file."""
        if not self.path:
            raise SystemError("File path is not set")
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(text)

    def tail(self, **kwargs):
        """Return the tail of the file."""
        return next(tail_file(self.path, **kwargs), "")

    def reversed(self, **kwargs):
        """Iterate over the file in reverse."""
        return tail_file(self.path, **kwargs)

    def head(self, **kwargs):
        """Return the first bytes of the file."""
        return next(head_file(self.path, **kwargs), "")

    def lines(self):
        """Iterate over the file lines."""
        if self.is_file():
            with open(self.path, "r", encoding="utf-8") as file:
                while line := file.readline():
                    yield line
        return ()

    def line_matching(self, word):
        """Return the first line matching the word."""
        return next((line for line in self.lines() if word in line), None)

    def last_line(self):
        """Return the last line of the file."""
        return last_line(self.path)

    def backup(self, tag, overwrite=False):
        """Create a backup copy of the file."""
        backup_file = self.path.with_name(f"{self.stem}{tag}{self.suffix}")
        if overwrite or not backup_file.exists():
            copy(self.path, backup_file)
            return backup_file

    def replace_text(self, text=(), pos=()):
        """Replace parts of the file text."""
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
        """Append raw bytes to the file."""
        with open(self.path, "ab") as file:
            file.write(data)

    def write_bytes(self, data):
        """Write bytes directly to disk."""
        with open(self.path, "wb") as file:
            file.write(data)
