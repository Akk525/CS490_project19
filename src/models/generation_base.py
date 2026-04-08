from abc import ABC, abstractmethod
from typing import List, Optional


class GenerationBase(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        raise NotImplementedError
