from cortecs_py.integrations.langchain.dedicated_llm import DedicatedLLM

__all__ = [
    'DedicatedLLM',
]

try:
    from cortecs_py.integrations.crewai.dedicated_crew import DedicatedCrew
    from cortecs_py.integrations.crewai.dedicated_crew_base import DedicatedCrewBase
    __all__.extend(['DedicatedCrewBase', 'DedicatedCrew'])
except ImportError:
    pass