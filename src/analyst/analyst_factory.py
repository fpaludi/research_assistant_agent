from langchain_core.language_models.chat_models import BaseChatModel
from analyst.analyst_agent import AnalystAgent


class AnalystAgentFactory:
    """Factory for creating AnalystAgent instances"""

    @staticmethod
    def create(llm: BaseChatModel, state_after_ok: str) -> AnalystAgent:
        """Create an AnalystAgent instance"""
        return AnalystAgent(llm=llm, state_after_ok=state_after_ok)
