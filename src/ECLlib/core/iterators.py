"""Iterator helpers used by :mod:`ECLlib`."""
from __future__ import annotations

import inspect


#===================================================================================================
class AutoRefreshIterator:                                                     # AutoRefreshIterator
#===================================================================================================
    """Iterator wrapper that refreshes itself when exhausted."""

    #-----------------------------------------------------------------------------------------------
    def __init__(self, iterable_factory, *args, **kwargs):                     # AutoRefreshIterator
    #-----------------------------------------------------------------------------------------------
        """Initialize the RefreshIterator.

        Args:
            iterable_factory: Callable producing an iterator supporting ``only_new``.
            *args: Positional arguments forwarded to ``iterable_factory``.
            **kwargs: Keyword arguments forwarded to ``iterable_factory``.
        """
        self._factory = iterable_factory
        params = inspect.signature(iterable_factory).parameters
        if "only_new" not in params:
            raise ValueError(
                f"Function {iterable_factory.__name__} does not support 'only_new' parameter."
            )
        kwargs["only_new"] = True
        self._iter = self._factory(*args, **kwargs)
        self._args = args
        self._kwargs = dict(kwargs)
        self._closed = False

    #-----------------------------------------------------------------------------------------------
    def __iter__(self):                                                        # AutoRefreshIterator
    #-----------------------------------------------------------------------------------------------
        """Return an iterator over the object."""
        return self

    #-----------------------------------------------------------------------------------------------
    def _refresh(self):                                                        # AutoRefreshIterator
    #-----------------------------------------------------------------------------------------------
        """Create a fresh underlying iterator from the factory."""
        if self._closed:
            return
        self._iter = self._factory(*self._args, **self._kwargs)

    #-----------------------------------------------------------------------------------------------
    def _close_current(self):                                                  # AutoRefreshIterator
    #-----------------------------------------------------------------------------------------------
        """Close and release the current underlying iterator exactly once."""
        current = self._iter
        self._iter = None
        close = getattr(current, "close", None)
        if close is not None:
            close()

    #-----------------------------------------------------------------------------------------------
    def close(self):                                                           # AutoRefreshIterator
    #-----------------------------------------------------------------------------------------------
        """Close the current iterator and prevent subsequent refreshes."""
        if self._closed:
            return
        self._closed = True
        self._close_current()

    #-----------------------------------------------------------------------------------------------
    def __next__(self):                                                        # AutoRefreshIterator
    #-----------------------------------------------------------------------------------------------
        """Return the next item from the iterator."""
        if self._closed:
            raise StopIteration
        refreshed = self._iter is None
        if refreshed:
            self._refresh()
        try:
            return next(self._iter)
        except StopIteration:
            self._close_current()
            if self._closed or refreshed:
                raise
            self._refresh()
            try:
                return next(self._iter)
            except StopIteration:
                self._close_current()
                raise
