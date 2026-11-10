"""Make the starting kit's loader work against the versions it pins.

``tracks/sleep_onset/benchmark_utils/nb_task.py`` (repo main, cloned
2026-10-30) sets ``neuro.infra.permissions: None`` in the data config, but
the ``neuralbench==0.4.0`` pinned by ``datasets/interaxon2026muse.py``
declares its config model with extras forbidden, so ``Data(**cfg)`` raises::

    ValidationError: 1 validation error for Data
    neuro.infra.permissions
      Extra inputs are not permitted

The key only sets filesystem permissions on the neuro cache, so dropping it
changes no data and no split. Patched here rather than in the clone so the
kit stays byte-identical to the official repo.
"""

import neuralbench.data as _nb_data

_DROP = (
    "neuro.infra.permissions",
    # ``NEURALBENCH_EXTRACT_CPUS`` (the kit's own switch for serial window
    # extraction) also writes ``target.infra.*``, a section this version of
    # the config model does not declare. The neuro extractor's own
    # ``cluster: None`` is what actually serialises the extraction, so
    # dropping this one keeps the switch usable.
    "target.infra",
)
_Real = _nb_data.Data


def _drop_path(mapping, path):
    """Remove a dotted key, whether it is flat or nested in sub-dicts."""
    if path in mapping:
        del mapping[path]
        return True
    head, _, tail = path.partition(".")
    child = mapping.get(head)
    if tail and isinstance(child, dict):
        return _drop_path(child, tail)
    return False


def _Patched(**kwargs):
    for key in _DROP:
        _drop_path(kwargs, key)
    return _Real(**kwargs)


def _serialise_exca_pool():
    """Make exca's caching backend run its shards in one worker thread.

    Window extraction goes through ``exca.steps.backends._PoolBackend``, which
    opens one process per core. Each child re-imports the neuro stack and so
    loads torch and its CUDA DLLs, and on a machine with a modest pagefile the
    pool dies with ``BrokenProcessPool`` (``WinError 1455``: the paging file is
    too small). A single-thread executor runs the same shards in the same
    order, in-process; extraction is bound by disk and MNE anyway.

    The kit's own ``NEURALBENCH_EXTRACT_CPUS`` switch does not reach this pool:
    it configures the neuro and target extractors, while this one belongs to
    the study source's cached backend.
    """
    from concurrent import futures

    from exca import utils as _exca_utils

    def _single_thread(pool, max_workers):  # noqa: ARG001
        return futures.ThreadPoolExecutor(max_workers=1)

    if getattr(_exca_utils.make_pool_executor, "__name__", "") != "_single_thread":
        _exca_utils.make_pool_executor = _single_thread
        import exca.steps.backends as _backends
        _backends.utils.make_pool_executor = _single_thread


def _add_pwrite():
    """Provide ``os.pwrite`` on Windows, where Python does not define it.

    ``exca.cachedict.core._record_use`` stamps a last-used timestamp into each
    cache entry with ``os.pwrite(fd, data, 0)``, a POSIX-only call. The
    positional write is equivalent to a seek plus a write here: nothing else
    in the process relies on the atomicity that ``pwrite`` would give, since
    the cache is being read by this one process.
    """
    import os

    if hasattr(os, "pwrite"):
        return False

    def pwrite(fd, data, offset):
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)

    os.pwrite = pwrite
    return True


def apply():
    """Install the patches; idempotent."""
    _add_pwrite()
    if getattr(_nb_data.Data, "__name__", "") != "_Patched":
        _nb_data.Data = _Patched
    _serialise_exca_pool()
    return _DROP
