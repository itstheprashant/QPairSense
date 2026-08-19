from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "QPairSense"
    app_version: str = "1.0.0"
    environment: str = "production"
    model_dir: str = str(BASE_DIR / "models")
    model_file: str = "model.joblib"
    vectorizer_file: str = "vectorizer.joblib"
    metadata_file: str = "metadata.json"
    max_question_length: int = 1000
    similarity_threshold: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
