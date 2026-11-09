"""
Shared HTTP etiquette for the free public APIs we call (Nominatim,
Wikimedia, ...).

Why this lives in one place: Nominatim's usage policy allows at most 1
request/second *across everything we send them*. Before this module, the
search agent and the web-evidence agent each kept their own timer, so a
search followed immediately by a reverse lookup could exceed the limit.
One shared throttle makes the policy true no matter which agent calls.
"""

import threading
import time

# Identifies the app to the services we call (generic defaults like
# "python-requests" are blocked by Nominatim). To add a contact address,
# change it here once and every agent picks it up.
USER_AGENT = "AccessMapAI/0.1 (college project; Pune accessibility assessment)"


class Throttle:
    """Blocks callers so that successive wait() returns are at least
    `min_interval` seconds apart. Thread-safe: FastAPI runs sync endpoints
    in a thread pool, so two requests can arrive at the same moment."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last = float("-inf")  # time.monotonic() of the last release

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self._min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


# Nominatim: max 1 request/second (https://operations.osmfoundation.org/policies/nominatim/)
nominatim_throttle = Throttle(1.0)