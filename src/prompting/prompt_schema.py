from pydantic import BaseModel


class PromptPayload(BaseModel):
    prompt_type: str
    raw_prompt: str
