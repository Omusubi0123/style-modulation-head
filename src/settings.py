from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    hf_token: str = Field(..., alias="HF_TOKEN")

    model_config = SettingsConfigDict(env_file=".env")


def get_settings():
    return Settings()


settings = get_settings()
