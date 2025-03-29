from langchain_core.language_models.chat_models import BaseChatModel
from report.report_agent import ReportGenerator


class ReportAgentFactory:
    """Factory for creating ReportGenerator instances"""

    @staticmethod
    def create(llm: BaseChatModel) -> ReportGenerator:
        """Create a ReportGenerator instance"""
        return ReportGenerator(llm=llm)
