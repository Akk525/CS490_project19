from abc import ABC, abstractmethod
from typing import List


class EmbedderBase(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
