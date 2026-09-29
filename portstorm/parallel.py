"""Map a function over tasks in worker processes, or in-process for a single worker."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed


def imap(fn, tasks, workers: int):
    if workers <= 1:
        for task in tasks:
            yield fn(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(fn, t) for t in tasks]):
            yield fut.result()
