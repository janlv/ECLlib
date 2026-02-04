"""Iterator helpers used by :mod:`ECLlib`."""
from __future__ import annotations

import inspect


#==================================================================================================
class AutoRefreshIterator:
#==================================================================================================
    """Iterator wrapper that refreshes itself when exhausted."""

    #----------------------------------------------------------------------------------------------
    def __init__(self, iterable_factory, *args, **kwargs):
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def __iter__(self):
    #----------------------------------------------------------------------------------------------
        """Return an iterator over the object."""
        return self

    #----------------------------------------------------------------------------------------------
    def _refresh(self):
    #----------------------------------------------------------------------------------------------
        """Create a fresh underlying iterator from the factory."""

        self._iter = self._factory(*self._args, **self._kwargs)

    #----------------------------------------------------------------------------------------------
    def __next__(self):
    #----------------------------------------------------------------------------------------------
        """Return the next item from the iterator."""
        try:
            return next(self._iter)
        except StopIteration:
            self._refresh()
            return next(self._iter)
