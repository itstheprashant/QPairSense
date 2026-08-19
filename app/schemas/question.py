from pydantic import BaseModel, Field, field_validator

class QuestionPairRequest(BaseModel):
    question1: str = Field(..., min_length=1, max_length=1000)
    question2: str = Field(..., min_length=1, max_length=1000)

    @field_validator("question1", "question2")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value

class PredictionResponse(BaseModel):
    is_duplicate: bool
    label: str
    confidence: float
    duplicate_probability: float
    non_duplicate_probability: float
    message: str
    model_version: str
