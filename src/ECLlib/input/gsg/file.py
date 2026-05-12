"""Public facade for Petrel/INTERSECT GSG property and grid files."""
from __future__ import annotations

from pathlib import Path

from .binary import (
    GSGFormatError,
    GSGMetadataError,
    UnsupportedGSGBlock,
    read_header,
    read_index_records,
)
from .grid import read_grid, write_grid_file
from .prop import find_property, read_properties, write_prop_file
from .types import (
    GSGGrid,
    GSGIndexEntry,
    GSGPillars,
    GSGProperty,
    entry_from_record,
)

__all__ = [
    "GSGFile",
    "GSGIndexEntry",
    "GSGProperty",
    "GSGGrid",
    "GSGPillars",
    "GSGFormatError",
    "GSGMetadataError",
    "UnsupportedGSGBlock",
]


#===================================================================================================
class GSGFile:                                                                             # GSGFile
#===================================================================================================
    """Read Petrel/INTERSECT ``.GSG`` files and write PROP or AXES/grid GSG files."""

    #-----------------------------------------------------------------------------------------------
    def __init__(self, path):                                                            # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Initialize the parser for ``path``."""
        self.path = Path(path)
        self.creator = ""
        self.version = ""
        self._data_start = 0
        self._format = None
        self._index = None
        with self.path.open("rb") as file_obj:
            self.creator, self.version, self._data_start = read_header(file_obj)
            self._index = tuple(
                entry_from_record(record)
                for record in read_index_records(file_obj)
            )
        if not self._index:
            raise GSGMetadataError("GSG file has no indexed blocks")
        self._format = self._index[0].key

    @property
    #-----------------------------------------------------------------------------------------------
    def format(self):                                                                    # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return the first indexed GSG keyword, usually ``PROP`` or ``AXES``."""
        return self._format

    #-----------------------------------------------------------------------------------------------
    def index(self):                                                                     # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return ordered index entries, preserving repeated keys."""
        if self._index is None:
            with self.path.open("rb") as file_obj:
                self._index = tuple(
                    entry_from_record(record)
                    for record in read_index_records(file_obj)
                )
        return self._index

    #-----------------------------------------------------------------------------------------------
    def blocks(self, key=None):                                                          # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Yield indexed block spans, optionally filtered by keyword."""
        for entry in self.index():
            if key is None or entry.key == key:
                yield entry

    #-----------------------------------------------------------------------------------------------
    def read(                                                                              # GSGFile
        self,
        *,
        kind=None,
        name=None,
        expand=True,
        shape="auto",
        read_areal=True,
        read_pillars=False,
        read_faults=False,
    ):
    #-----------------------------------------------------------------------------------------------
        """Read GSG content using a format-aware convenience API."""
        read_kind = self.format.casefold() if kind is None else str(kind).casefold()
        if name is not None:
            if read_kind not in {"prop", "properties"}:
                raise GSGFormatError("name can only be used when reading PROP properties")
            if shape == "auto":
                shape = self._property_shape()
            return self.property(name, expand=expand, shape=shape)
        if read_kind in {"prop", "properties"}:
            if shape == "auto":
                shape = self._property_shape()
            return tuple(self.properties(expand=expand, shape=shape))
        if read_kind in {"axes", "grid"}:
            return self.grid(
                read_areal=read_areal,
                read_pillars=read_pillars,
                read_faults=read_faults,
            )
        if read_kind == "index":
            return self.index()
        if read_kind == "blocks":
            return tuple(self.blocks())
        raise ValueError(f"Unsupported GSG read kind {kind!r}")

    #-----------------------------------------------------------------------------------------------
    def dim(self):                                                                       # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return grid dimensions from AXES metadata or the adjacent INTERSECT case."""
        if self.format == "AXES":
            return self.grid(read_areal=False, read_pillars=False).dimensions

        afi_files = tuple(self.path.parent.glob("*.afi"))
        matches = tuple(
            path for path in afi_files
            if self.path.stem == path.stem or self.path.stem.startswith(f"{path.stem}_")
        )
        if matches:
            afi_files = (max(matches, key=lambda path: len(path.stem)),)
        if len(afi_files) != 1:
            raise GSGMetadataError(
                f"Expected one matching AFI file beside {self.path}, found {len(afi_files)}"
            )

        from ..intersect import IX_input

        try:
            dimensions = IX_input(afi_files[0]).dim()
        except Exception as exc:
            raise GSGMetadataError(f"Could not read grid dimensions from {afi_files[0]}") from exc
        if not dimensions:
            raise GSGMetadataError(f"Could not read grid dimensions from {afi_files[0]}")
        return tuple(int(value) for value in dimensions)

    #-----------------------------------------------------------------------------------------------
    def properties(self, expand=True, shape=None):                                       # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Yield decoded PROP records as :class:`GSGProperty` objects."""
        if self.format != "PROP":
            raise GSGFormatError(f"properties() requires a PROP GSG file, got {self.format!r}")
        yield from read_properties(self.path, self.index(), expand=expand, shape=shape)

    #-----------------------------------------------------------------------------------------------
    def property(self, alias_or_name, expand=True, shape=None):                          # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return one property matched by alias or CASE_PROPS name."""
        return find_property(
            self.path,
            self.index(),
            alias_or_name,
            expand=expand,
            shape=shape,
        )

    @staticmethod
    #-----------------------------------------------------------------------------------------------
    def write_prop(                                                                       # GSGFile
        path,
        *properties,
        creator="PetrelForIx",
        version="2022.9.0",
        overwrite=False,
    ):
    #-----------------------------------------------------------------------------------------------
        """Write a PROP GSG file and return the output path."""
        return write_prop_file(
            path,
            *properties,
            creator=creator,
            version=version,
            overwrite=overwrite,
        )

    @staticmethod
    #-----------------------------------------------------------------------------------------------
    def write_grid(                                                                       # GSGFile
        path,
        grid,
        *,
        creator="PetrelForIx",
        version="2022.9.0",
        overwrite=False,
    ):
    #-----------------------------------------------------------------------------------------------
        """Write an AXES/grid GSG file and return the output path."""
        return write_grid_file(
            path,
            grid,
            creator=creator,
            version=version,
            overwrite=overwrite,
        )

    #-----------------------------------------------------------------------------------------------
    def grid(self, read_areal=True, read_pillars=False, read_faults=False):              # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return decoded AXES grid metadata and optional payload arrays."""
        if self.format != "AXES":
            raise GSGFormatError(f"grid() requires an AXES GSG file, got {self.format!r}")
        return read_grid(
            self.path,
            self.index(),
            read_areal=read_areal,
            read_pillars=read_pillars,
            read_faults=read_faults,
        )

    #-----------------------------------------------------------------------------------------------
    def _property_shape(self):                                                           # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return dimensions for automatic PROP shaping, or ``None`` if unavailable."""
        try:
            return self.dim()
        except GSGMetadataError:
            return None
