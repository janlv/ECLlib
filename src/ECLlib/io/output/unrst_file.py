"""
Eclipse/INTERSECT UNRST output file handler.

This module contains the full UNRST_file implementation together with
UNRST-specific read and write helpers.
"""

from collections import namedtuple
from datetime import datetime
from fnmatch import fnmatch
from itertools import islice

from numpy import array as nparray, sum as npsum

from ...core import AutoRefreshIterator, File
from ...utils import flatten, flatten_all, remove_chars
from ..unformatted.base import unfmt_block, unfmt_file

__all__ = ["UNRST_file"]


#===================================================================================================
class UNRST_file(unfmt_file):                                                          # UNRST_file
#===================================================================================================
    """
    Reader for Eclipse UNRST restart files.

    UNRST (Unified Restart File)
    Binary unformatted file containing solution data arrays for all active grid cells
    at each report step. Includes pressure, phase saturations, and other cell-based variables,
    as well as optional well, group, and non-neighbor connection (NNC) data.
    """

    start = "SEQNUM"
    end = "ENDSOL"
    #           variable   keyword   position (None = whole array)
    var_pos = {"step": ("SEQNUM", 0),
               "nx": ("INTEHEAD", 8),
               "ny": ("INTEHEAD", 9),
               "nz": ("INTEHEAD", 10),
               "nwell": ("INTEHEAD", 16),
               "day": ("INTEHEAD", 64),
               "month": ("INTEHEAD", 65),
               "year": ("INTEHEAD", 66),
               "hour": ("INTEHEAD", 206),
               "min": ("INTEHEAD", 207),
               "sec": ("INTEHEAD", 410),
               "time": ("DOUBHEAD", 0),
               "wells": ("ZWEL", None)}  # No ZWEL in first section

    #-----------------------------------------------------------------------------------------------
    def __init__(self, file, suffix=".UNRST", end=None, role=None):                    # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Initialize the UNRST_file."""
        super().__init__(file, suffix=suffix, role=role)
        self.end = end or self.end
        self._dim = None
        self._units = None

    #-----------------------------------------------------------------------------------------------
    def __len__(self):                                                                 # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the number of entries."""
        return len(list(self.steps()))

    #-----------------------------------------------------------------------------------------------
    def dim(self):                                                                     # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the grid dimensions."""
        self._dim = self._dim or next(self.read("nx", "ny", "nz"))
        return self._dim

    #-----------------------------------------------------------------------------------------------
    def _check_for_missing_keys(self, *in_keys, keys=None):                            # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Verify that required keywords are present."""
        keys = keys or self.find_keys(*in_keys)
        if missing := [ik for ik in in_keys if not any(fnmatch(k, ik) for k in keys)]:
            raise ValueError(f"The following keywords are missing in {self}: {missing}")
        return keys

    #-----------------------------------------------------------------------------------------------
    def _cellnr(self, coord, base=0):                                                  # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """
        Return position in 1D array given 3D coordinate of base=0 (default) or base=1
        """
        dim = self.dim()
        # Apply negative index from the end
        coord = [c if c >= 0 else dim[i] + c + base for i, c in enumerate(coord)]
        return base + coord[0] - base + dim[0] * (coord[1] - base) + dim[0] * dim[1] * (coord[2] - base)

    #-----------------------------------------------------------------------------------------------
    def celldata(self, coord, *keywords, base=0):                                      # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """
        Return the given keywords as a celldata namedtuple for the given cell-coordinate.

        Keyword arguments
            base     : Zero- or one-based indexing (0 is default)
            time_res : Time resolution, valid values are 'day', 'hour', 'min', 'sec' ('day' is default)
        """
        self._check_for_missing_keys(*keywords)
        cellnr = self._cellnr(coord, base=base)
        args = flatten((key, cellnr) for key in keywords)
        data = zip(*self.blockdata(*args, singleton=False))
        celldata = namedtuple("celldata", ("days",) + keywords)
        return celldata(tuple(self.days()), *data)

    #-----------------------------------------------------------------------------------------------
    def cellarray(self, *in_keys, start=None, stop=None, step=1, warn_missing=True):   # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return cell array data."""
        step = step or self.count_sections()
        keys = self.find_keys(*in_keys)
        if warn_missing:
            self._check_for_missing_keys(*in_keys, keys=keys)
        names = [remove_chars("+-", k) for k in keys]
        celltuple = namedtuple("cellarray", ["days", "date"] + names)
        dds = zip(self.days(), self.dates(), self.section_blocks())
        for day, date, section in islice(dds, start, stop, step):
            blockdata = {k: None for k in keys}
            for block in section:
                if (key := block.key()) in keys:
                    blockdata[key] = block.data()
            yield celltuple(day, date, *self.reshape_dim(*blockdata.values()))

    #-----------------------------------------------------------------------------------------------
    def wells(self, stop=None):                                                        # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the wells defined in the file."""
        wells = flatten_all(islice(self.read("wells"), 0, stop))
        unique_wells = set(w for well in wells if (w := well.strip()))
        return tuple(unique_wells)

    #-----------------------------------------------------------------------------------------------
    def open_wells(self):                                                              # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return wells that currently produce."""
        for ihead, icon in self.blockdata("INTEHEAD", "ICON"):
            niconz, ncwmax, nwells = ihead[32], ihead[17], ihead[16]
            icon = nparray(icon).reshape((niconz, ncwmax, nwells), order="F")
            yield sum(npsum(icon[5, :, :], axis=0) > 0)

    #-----------------------------------------------------------------------------------------------
    def steps(self):                                                                   # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the simulation report steps."""
        return flatten_all(self.read("step"))

    #-----------------------------------------------------------------------------------------------
    def end_step(self):                                                                # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the final report step."""
        return self.last_value("step")

    #-----------------------------------------------------------------------------------------------
    def end_time(self):                                                                # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the final timestamp."""
        return self.last_value("time")

    #-----------------------------------------------------------------------------------------------
    def end_date(self):                                                                # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the last simulation date."""
        return next(self.dates(tail=True), None)

    #-----------------------------------------------------------------------------------------------
    def dates(self, resolution="day", **kwargs):                                       # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the simulation dates."""
        varnames = ("year", "month", "day")
        if resolution == "day":
            pass
        elif resolution == "hour":
            varnames += ("hour",)
        elif resolution == "min":
            varnames += ("hour", "min")
        elif resolution == "sec":
            varnames += ("hour", "min", "sec")
            # Seconds are reported as microseconds, integer-divide by 1e6
            return (datetime(*vars[:-1], int(vars[-1] // 1e6)) for vars in self.read(*varnames, **kwargs))
        else:
            raise SyntaxError("resolution must be 'hour', 'min', or 'sec'")
        return (datetime(*vars) for vars in self.read(*varnames, **kwargs))

    #-----------------------------------------------------------------------------------------------
    def units(self):                                                                   # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return unit names per keyword."""
        if self._units is None:
            ihead2 = next(self.blockdata("INTEHEAD", 2), None)
            if ihead2:
                self._units = {1: "metric", 2: "field", 3: "lab", 4: "pvt-m"}[ihead2]
        return self._units

    #-----------------------------------------------------------------------------------------------
    def days(self, **kwargs):                                                          # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the simulation days."""
        convert = 1
        if self.units() == "lab":
            # DOUBHEAD[0] is given in hours in lab units
            convert = 1 / 24
        return (next(flatten(dh)) * convert for dh in self.blockdata("DOUBHEAD", singleton=True, **kwargs))

    #-----------------------------------------------------------------------------------------------
    def section(self, days=None, date=None):                                           # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return a named section."""
        stop = None
        if days:
            data_func = self.days
            stop = days
            return next(i for i, val in enumerate(self.days()) if val >= days)
        if date:
            data_func = self.dates
            stop = datetime(*date)
        if not stop:
            raise ValueError("Either days or date must be given")
        return next(i for i, val in enumerate(data_func()) if val >= stop)

    #-----------------------------------------------------------------------------------------------
    def merge_keys_from(self, donor, *, keys, name=None, rename=None,                 # UNRST_file
                        overwrite=False):
    #-----------------------------------------------------------------------------------------------
        """Write a merged UNRST file that appends selected donor keys to each matching section."""
        self.exists(raise_error=True)
        donor_file = donor if isinstance(donor, UNRST_file) else UNRST_file(donor)
        donor_file.exists(raise_error=True)
        keyset = set(keys)
        rename_map = {} if rename is None else dict(rename)
        outfile = File(name or self.path.parent / f"{self.path.stem}_MERGED.UNRST", suffix=".UNRST")
        if outfile.path in {self.path, donor_file.path}:
            raise ValueError(
                "UNRST_file.merge_keys_from() does not support writing to the host or donor path"
            )
        if outfile.exists() and not overwrite:
            raise FileExistsError(f"{outfile} already exists; pass overwrite=True to replace it")
        outfile.parent.mkdir(parents=True, exist_ok=True)
        host_endblock = self.end_key() or self.end
        donor_endblock = donor_file.end_key() or donor_file.end
        try:
            with open(outfile.path, "wb") as out:
                host_sections = _iter_host_sections(self, host_endblock)
                donor_sections = _iter_donor_sections(donor_file, keyset, rename_map, donor_endblock)
                for (_, host_prefix, host_end), (_, donor_blocks) in zip(host_sections, donor_sections, strict=True):
                    _write_merged_section(out, host_prefix=host_prefix, donor_blocks=donor_blocks, host_end=host_end)
        except Exception:
            outfile.delete()
            raise
        return UNRST_file(outfile.path, end=host_endblock)

    #-----------------------------------------------------------------------------------------------
    def append_blocks(self, *, step, keys, blocks, dtypes=None, endblock="ENDSOL"):    # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Append one row of new blocks before the end marker in the last section."""
        self.exists(raise_error=True)
        endblock = str(endblock).strip()
        keyset = tuple(keys)
        payloads = tuple(blocks)

        step = int(step)
        last_step = int(self.end_step())
        if step != last_step:
            raise ValueError(
                f"UNRST_file.append_blocks() only supports the last section; "
                f"expected step {last_step}, got {step}"
            )
        end_marker = next(self.tail_blocks(), None)
        if end_marker is None or end_marker.key() != endblock:
            raise ValueError(f"{self} does not end with {endblock}")

        payload = b"".join(
            unfmt_block.from_data(key, block, dtype).as_bytes()
            for key, block, dtype in _iter_block_payloads(keyset, payloads, dtypes)
        )
        payload += self.binarydata(pos=(end_marker.startpos, end_marker.endpos), raise_error=True)
        self.resize(start=end_marker.startpos, end=end_marker.endpos)
        self.append_bytes(payload)
        self.end = endblock
        return self

    #-----------------------------------------------------------------------------------------------
    def section_copy_slices(self, keys=None, only_new=False):                          # UNRST_file
    #-----------------------------------------------------------------------------------------------
        # Yield header and optional data slices for each section.
        keys = tuple(keys or ())
        keyset = set(keys)
        for section in self.section_blocks(only_new=only_new):
            step = -1
            header_start = None
            header_slice = None
            data_slices = []
            in_data = False
            for block in section:
                if self.start in block:
                    step = block.data()[0]
                    header_start = block.startpos
                if block.key() == "STARTSOL":
                    header_slice = slice(header_start, block.endpos)
                    in_data = True
                    continue
                if block.key() == "ENDSOL":
                    break
                if in_data and keyset and block.key() in keyset:
                    data_slices.append(slice(block.startpos, block.endpos))
            yield (step, header_slice, tuple(data_slices))

    #-----------------------------------------------------------------------------------------------
    def section_copy_iter(self, keys=None, only_new=False):                            # UNRST_file
    #-----------------------------------------------------------------------------------------------
        # Return a AutoRefreshIterator over section_copy_slices.
        #
        # Note:
        #   The AutoRefresIterator is useful for the UNRST-files that are actively written
        #   since the iterator automatically refreshes when exhausted to capture new sections
        #   written after the iterator was created.
        return AutoRefreshIterator(self.section_copy_slices, keys, only_new=only_new)

    #-----------------------------------------------------------------------------------------------
    def end_key(self):                                                                 # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return the keyword used to terminate the block."""
        block = next(self.tail_blocks(), None)
        if block:
            return block.key()

    #-----------------------------------------------------------------------------------------------
    def from_Xfile(self, xfile, log=False, delete=False):                              # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """
        Append a SEQNUM block at the beginning of the non-unified restart X-file.
        """
        xfile = File(xfile)
        if not xfile.exists():
            raise FileNotFoundError(f"{xfile} is missing")
        # Add missing SEQNUM at beginning
        step = int(xfile.suffix[-4:])
        seqnum = unfmt_block.from_data("SEQNUM", [step], "int")
        self.merge([(step, seqnum.as_bytes())], [(step, xfile.binarydata())])
        if delete:
            xfile.delete(raise_error=True)
            if log:
                log(f"Deleted {xfile}")
        if callable(log):
            log(f"Created {self} from {xfile}")

    #-----------------------------------------------------------------------------------------------
    def as_Xfiles(self, log=False, stop=None):                                         # UNRST_file
    #-----------------------------------------------------------------------------------------------
        """Return helper objects mirroring Eclipse X files."""
        for i, sec in enumerate(self.section_blocks()):
            xfile = self.with_suffix(f".X{i:04d}")
            with open(xfile, "wb") as file:
                for block in sec:
                    key = block.key()
                    if key != "SEQNUM":
                        file.write(block.binarydata())
                    if key == "ENDSOL":
                        break
            if callable(log):
                log(f"Wrote {xfile}")
            if stop and i == stop:
                return


#---------------------------------------------------------------------------------------------------
def _infer_block_dtype(payload):
#---------------------------------------------------------------------------------------------------
    """Infer a block dtype name from a NumPy payload."""
    kind = payload.dtype.kind
    if kind == "b":
        return "bool"
    if kind in "iu":
        return "int"
    if kind == "f":
        return "float" if payload.dtype.itemsize <= 4 else "double"
    if kind in "SU":
        return "char"
    raise TypeError(f"Unable to infer block dtype from numpy dtype {payload.dtype!s}")


#---------------------------------------------------------------------------------------------------
def _iter_block_payloads(keys, blocks, dtypes):
#---------------------------------------------------------------------------------------------------
    """Yield append payloads aligned with their keys and effective dtypes."""
    payload_types = None if dtypes is None else tuple(dtypes)
    if payload_types is None:
        for key, block in zip(tuple(keys), tuple(blocks), strict=True):
            yield key, block, _infer_block_dtype(block)
        return
    yield from zip(tuple(keys), tuple(blocks), payload_types, strict=True)


#---------------------------------------------------------------------------------------------------
def _iter_host_sections(unrst, endblock):
#---------------------------------------------------------------------------------------------------
    """Yield host section prefix bytes and terminator bytes using mmap slices."""
    section_start = None
    step = -1
    for block in unrst.blocks(use_mmap=True):
        key = block.key()
        if key == unrst.start:
            section_start = block.startpos
            step = int(block.data()[0])
            continue
        if key == endblock:
            yield step, block._data[section_start:block.startpos], block.binarydata()


#---------------------------------------------------------------------------------------------------
def _iter_donor_sections(unrst, keys, rename_map, endblock):
#---------------------------------------------------------------------------------------------------
    """Yield selected donor solution block bytes for each section."""
    in_solution = False
    step = -1
    blocks = []
    for block in unrst.blocks(use_mmap=True):
        key = block.key()
        if key == unrst.start:
            step = int(block.data()[0])
            in_solution = False
            blocks = []
            continue
        if key == "STARTSOL":
            in_solution = True
            continue
        if in_solution and key == endblock:
            yield step, tuple(blocks)
            in_solution = False
            blocks = []
            continue
        if in_solution and key in keys:
            out_key = rename_map.get(key, key)
            blocks.append(block.binarydata() if out_key == key else block.renamed_bytes(out_key))


#---------------------------------------------------------------------------------------------------
def _write_merged_section(out, *, host_prefix, donor_blocks, host_end):
#---------------------------------------------------------------------------------------------------
    """Write one merged section from raw host slices and donor block bytes."""
    out.write(host_prefix)
    for data in donor_blocks:
        out.write(data)
    out.write(host_end)
