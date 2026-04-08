from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ProceduralExample(BaseModel):
    example_id: str
    source_dataset: str
    source_item_id: str
    domain: str
    image_path: Optional[str] = None
    goal: str
    full_procedure: List[str]
    current_state: str
    disrupted_step_index: int
    disrupted_step_text: str
    disruption_type: str
    disruption_description: str
    available_context: Dict[str, Any] = Field(default_factory=dict)
    target_adaptation: Optional[str] = None
    target_provenance: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('goal')
    @classmethod
    def validate_goal(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('goal cannot be empty')
        return value

    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator('full_procedure')
    @classmethod
    def validate_steps(cls, value: List[str]) -> List[str]:
        if len(value) < 3:
            raise ValueError('full_procedure requires at least 3 steps')
        return value
