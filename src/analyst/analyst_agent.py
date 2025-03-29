from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END
from analyst.io_models import GenerateAnalystsState, Perspectives



class AnalystAgent:
    ANALYST_INSTRUCTIONS="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

    def __init__(
        self,
        llm: BaseChatModel,
        state_after_ok: str,
    ):
        """Initialize the analyst agent with a language model."""
        self.llm = llm
        self.structured_llm = llm.with_structured_output(Perspectives)
        self.state_after_ok = state_after_ok

    def create_analysts(self, state: GenerateAnalystsState) -> dict:
        """Create analysts based on the provided state."""

        topic = state['topic']
        max_analysts = state['max_analysts']
        human_analyst_feedback = state.get('human_analyst_feedback', '')

        # System message
        system_message = self.ANALYST_INSTRUCTIONS.format(
            topic=topic,
            human_analyst_feedback=human_analyst_feedback,
            max_analysts=max_analysts
        )

        # Generate analysts
        analysts: Perspectives = self.structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts.")
        ])

        # Return analysts in state format
        return {"analysts": analysts.analysts}

    def should_continue(self, state: GenerateAnalystsState):
        """ Return the next node to execute """

        # Check if human feedback
        human_analyst_feedback=state.get('human_analyst_feedback', None)
        if human_analyst_feedback:
            return "create_analysts"

        # Otherwise continue
        return self.state_after_ok

    def human_feedback(self, state: GenerateAnalystsState):
        """ No-op node that should be interrupted on """
        pass
