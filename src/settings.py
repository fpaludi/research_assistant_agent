from pydantic import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str
    LANGSMITH_API_KEY: str
    LANGCHAIN_API_KEY: str


settings = Settings()
