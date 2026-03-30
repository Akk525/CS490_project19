from abc import ABC, abstractmethod
from typing import List, Tuple


class RerankerBase(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        raise NotImplementedError
