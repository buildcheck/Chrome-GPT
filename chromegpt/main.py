from chromegpt.agent.autogpt import AutoGPTAgent
from chromegpt.agent.chromegpt_agent import ChromeGPTAgent
from chromegpt.agent.zeroshot import BabyAGIAgent, ZeroShotAgent


def run_chromegpt(
    task: str,
    model: str = "gpt-3.5-turbo",
    agent: str = "zero-shot",
    headless: bool = False,
    verbose: bool = False,
    continuous: bool = True,
    max_iterations: int = 5,
) -> str:
    """Run ChromeGPT."""
    # setup agent
    if agent == "auto-gpt":
        agent_obj: ChromeGPTAgent = AutoGPTAgent(
            model=model, verbose=verbose, continuous=continuous
        )
    elif agent == "baby-agi":
        agent_obj = BabyAGIAgent(model=model, verbose=verbose)
    elif agent == "zero-shot":
        agent_obj = ZeroShotAgent(model=model, verbose=verbose, max_iterations=max_iterations)
    else:
        raise ValueError(f"Agent {agent} not found.")
    # run agent
    return agent_obj.run([task])


if __name__ == "__main__":
    TASK_TEXT = ""
    run_chromegpt(
        task=TASK_TEXT,
        model='gpt-4o',
        agent="zero-shot",
        max_iterations=4)
