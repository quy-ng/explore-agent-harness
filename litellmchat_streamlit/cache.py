from collections import OrderedDict
import time
from typing import Optional


class LRUCache:
    """A tiny in-memory LRU cache for string values.

    Not thread-safe. Stores (value, timestamp) tuples.
    """

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._data = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def get(self, key: str) -> Optional[str]:
        item = self._data.get(key)
        if item is None:
            return None
        value, ts = item
        # mark as recently used
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        self._data[key] = (value, time.time())
        # mark as recently used
        self._data.move_to_end(key)
        self._evict_if_needed()

    def clear(self) -> None:
        self._data.clear()
