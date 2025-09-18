"""Filesystem related utilities."""
from __future__ import annotations

from mmap import ACCESS_READ, mmap
from pathlib import Path
from shutil import copy2, rmtree

import stat
from re import compile as re_compile

from .string_ops import decode, upper_and_lower


def empty_folder(folder: str | Path) -> Path:
    """Ensure ``folder`` exists and is empty."""
    folder = Path(folder)
    if folder.exists():
        rmtree(folder)
    folder.mkdir()
    return folder


def has_write_access(path: str | Path, error: str | bool = False) -> bool:
    """Return ``True`` if ``path`` is writable."""
    path = Path(path)
    name = "write.access"
    test = path.with_name(name) if path.is_file() else path / name
    try:
        test.touch()
    except PermissionError as perm_err:
        if error:
            raise SystemError(error) from perm_err
        return False
    test.unlink()
    return True


def make_user_executable(path: str | Path) -> None:
    """Grant the execute bit for the owner of ``path``."""
    path = Path(path)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def copy_recursive(src: str | Path, dst: str | Path, log=None) -> None:
    """Copy ``src`` recursively to ``dst``."""
    src = Path(src)
    dst = Path(dst)
    if src.is_file():
        src_items = (src,)
    else:
        src_items = tuple(src.iterdir())
    if dst.is_file():
        dst_items = (dst,)
        dst = dst.parent
    else:
        dst_items = (dst / item.name for item in src_items)
    dst.mkdir(exist_ok=True, parents=True)
    for _src, _dst in zip(src_items, dst_items):
        if _src.is_file():
            copy2(_src, _dst)
            if log:
                log(f"Copied {_src} -> {_dst}")
        else:
            copy_recursive(_src, _dst, log=log)


def last_line(path: str | Path) -> str:
    """Return the last line of ``path``."""
    with open(path, "rb") as file:
        try:
            file.seek(-2, 2)
            while file.read(1) != b"\n":
                file.seek(-2, 1)
        except OSError:
            file.seek(0)
        return file.readline().decode()


def tail_file(path: str | Path, size: int = 10 * 1024, size_limit: bool = False):
    """Yield ``size`` byte chunks from the end of ``path``."""
    path = Path(path)
    if not path.is_file():
        return
    filesize = path.stat().st_size
    if size_limit and filesize < size:
        return
    size = pos = min(size, filesize)
    with open(path, "rb") as file:
        while pos <= filesize:
            file.seek(-pos, 2)
            yield decode(file.read(size))
            if pos == filesize:
                return
            pos = min(pos + size, filesize)


def head_file(path: str | Path, size: int = 10 * 1024):
    """Yield ``size`` byte chunks from the start of ``path``."""
    path = Path(path)
    if not path.is_file():
        return
    with open(path, "rb") as file:
        while data := file.read(size):
            yield decode(data)


def file_exists(file: str | Path, raise_error: bool = False) -> bool:
    """Return ``True`` if ``file`` exists."""
    if Path(file).is_file():
        return True
    if raise_error:
        raise SystemError(f"ERROR {Path(file).name} does not exist")
    return False


def file_not_empty(file: str | Path) -> bool:
    """Return ``True`` if ``file`` exists and is not empty."""
    path = Path(file)
    return path.is_file() and path.stat().st_size > 0


def print_file(file: str | Path) -> None:
    """Print the contents of ``file`` to stdout."""
    with open(file) as f:
        lines = f.readlines()
    print("".join(lines))


def read_line(n: int, file: str | Path, raise_error: bool = True) -> str:
    """Return line ``n`` from ``file``."""
    file = Path(file)
    if not file.is_file():
        if raise_error:
            raise SystemError(f"ERROR {file} not found in read_line()")
        return ""
    with open(file, "r") as fileobj:
        for _ in range(n - 1):
            fileobj.readline()
        return fileobj.readline()


def read_file(file: str | Path, raise_error: bool = True, skip: int | None = None) -> str:
    """Return the entire contents of ``file``."""
    file = Path(file)
    if not file.is_file():
        if raise_error:
            raise SystemError(f"ERROR {file} not found in read_file()")
        return ""
    with open(file, "rb") as fileobj:
        if skip:
            fileobj.seek(skip)
        return decode(fileobj.read())


def write_file(path: str | Path, text: str) -> int:
    """Write ``text`` to ``path`` using UTF-8 or Latin-1 encoding."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "w", encoding=encoding) as file:
                return file.write(text)
        except UnicodeError:
            continue
    raise SystemError(f"ERROR Unable to write to file {Path(path).name}")


def remove_comments(path: str | Path, comment: str = "--", join: bool = True, raise_error: bool = True):
    """Return ``path`` stripped of lines starting with ``comment``."""
    path = Path(path)
    try:
        if not path.is_file():
            if raise_error:
                raise SystemError(f"ERROR {path} not found in remove_comments()")
            return []
    except PermissionError:
        return []
    comment_bytes = comment.encode()
    with open(path, "rb") as file:
        data = file.read()
        lines = (
            line.split(comment_bytes)[0].strip()
            for l in data.split(b"\n")
            if (line := l.strip()) and not line.startswith(comment_bytes)
        )
        if join:
            return decode(b"\n".join(lines)) + "\n"
        return [decode(line) for line in lines]


def is_file_ignore_suffix_case(file: str | Path):
    """Return the existing file path ignoring suffix case."""
    file = Path(file)
    files = [file.with_suffix(ext) for ext in upper_and_lower([file.suffix])]
    for candidate in files:
        if candidate.is_file():
            return candidate
    return False


def replace_line(fname: str | Path, find: str | None = None, replace: str | None = None) -> bool:
    """Replace ``find`` with ``replace`` in ``fname``."""
    if not Path(fname).is_file():
        return False
    with open(fname, "r") as file:
        lines = file.readlines()
    positions = [i for i, line in enumerate(lines) if find in line]
    if positions:
        lines[positions[0]] = replace
    else:
        lines.append(replace)
    with open(fname, "w") as file:
        file.write("".join(lines))
    return True


def delete_all(folder: str | Path, keep_folder: bool = False, ignore_error: tuple = ()) -> None:
    """Delete the contents of ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    for child in folder.iterdir():
        try:
            if child.is_file():
                child.unlink()
            else:
                delete_all(child)
        except ignore_error:
            pass
    if not keep_folder:
        folder.rmdir()


def silentdelete(*fname: str | Path, echo: bool = False) -> None:
    """Delete provided filenames, ignoring missing ones."""
    for name in fname:
        file = Path(name)
        try:
            if file.is_file():
                file.unlink()
                if echo:
                    print(f"Deleted {name}")
        except (PermissionError, FileNotFoundError) as err:
            if echo:
                print(f"Unable to delete {name}: {err}")


def delete_files_matching(*pattern: str | Path, echo: bool = False, raise_error: bool = False) -> str:
    """Delete files matching the provided glob patterns."""
    message = ""
    for pat in pattern:
        pat = Path(pat)
        for file in pat.parent.glob(pat.name):
            if echo:
                print("Removing " + str(file))
            try:
                file.unlink()
            except PermissionError as err:
                message = "WARNING Unable to delete file " + str(file) + ", maybe it belongs to another process"
                if raise_error:
                    raise SystemError(message) from err
    return message


def safeopen(filename: str | Path, mode: str):
    """Open ``filename`` and raise a helpful error on failure."""
    try:
        filehandle = open(filename, mode)
        return filehandle
    except OSError as error:
        raise SystemError(f"Unable to open file {filename}: {error}") from error


def warn_empty_file(file: str | Path, comment: str = "") -> None:
    """Warn if ``file`` contains no non-comment content."""
    with open(file, "r") as f:
        for line in f:
            if not line.startswith(comment) and not line.isspace():
                return
    print(f"WARNING! {file} is empty")


def matches(
    file: str | Path | None = None,
    pattern: str | None = None,
    length: int = 0,
    multiline: bool = False,
    pos: int | None = None,
    check: bytes | None = None,
):
    """Yield regex matches from ``file`` using memory mapping."""
    path = Path(file) if file is not None else None
    if path is None or not path.is_file() or path.stat().st_size < 1:
        return []
    flags = 0
    if multiline:
        from re import DOTALL

        flags = DOTALL
    regexp = re_compile(pattern.encode(), flags=flags)
    with open(path) as f:
        with mmap(f.fileno(), length=length, access=ACCESS_READ) as data:
            if pos:
                data = data[pos:]
            if check and check not in data:
                data = b""
            yield from regexp.finditer(data)


def number_of_blocks(file: str | Path | None = None, blockstart: str | None = None) -> int:
    """Return the number of blocks in ``file`` using ``blockstart`` as delimiter."""
    prev = None
    for match in matches(file=file, pattern=blockstart):
        if prev is not None:
            blocksize = match.start() - prev
            break
        prev = match.start()
    else:
        blocksize = 1
    return round(Path(file).stat().st_size / blocksize)


def count_match(file: str | Path | None = None, pattern: str | None = None) -> int:
    """Return the number of matches of ``pattern`` in ``file``."""
    with open(file) as f:
        wholefile = mmap(f.fileno(), 0, access=ACCESS_READ)
        from re import findall

        return len(findall(pattern.encode(), wholefile))


__all__ = [
    "count_match",
    "copy_recursive",
    "delete_all",
    "delete_files_matching",
    "empty_folder",
    "file_exists",
    "file_not_empty",
    "head_file",
    "has_write_access",
    "is_file_ignore_suffix_case",
    "last_line",
    "make_user_executable",
    "matches",
    "number_of_blocks",
    "print_file",
    "read_file",
    "read_line",
    "remove_comments",
    "replace_line",
    "safeopen",
    "silentdelete",
    "tail_file",
    "warn_empty_file",
    "write_file",
]
