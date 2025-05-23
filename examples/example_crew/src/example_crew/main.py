#!/usr/bin/env python
import sys

from example_crew.crew import ExampleCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding necessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run() -> None:
    """
    Run the crew.
    """
    inputs = {"topic": "AI LLMs"}
    ExampleCrew().crew().kickoff(inputs=inputs)


def train() -> None:
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        ExampleCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def replay() -> None:
    """
    Replay the crew execution from a specific task.
    """
    try:
        ExampleCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test() -> None:
    """
    Test the crew execution and returns the results.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        ExampleCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


if __name__ == "__main__":
    run()
