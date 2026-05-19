from abc import ABC, abstractmethod
from typing import List


class BaseScheduler(ABC):
    """Minimal abstract base class for all schedulers."""

    @abstractmethod
    def schedule(self, requests: List) -> List:
        """Accept a list of requests and return a sorted/filtered list."""
