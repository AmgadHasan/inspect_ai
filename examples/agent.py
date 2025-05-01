import time
from typing import Generator

from inspect_ai import Task, eval, task
from inspect_ai.agent._agent import AgentState, agent
from inspect_ai.agent._run import run
from inspect_ai.dataset import Sample
from inspect_ai.model._model import get_model
from inspect_ai.model._model_output import ModelOutput, ModelUsage
from inspect_ai.scorer import exact
from inspect_ai.solver._solver import Generate, solver
from inspect_ai.solver._task_state import TaskState
from inspect_ai.util._limit import (
    LimitExceededError,
    time_limit,
    token_limit,
    working_time_limit,
)


@solver
def my_solver():
    async def solve(state: TaskState, generate: Generate):
        await generate(state)

        try:
            results = await run(my_agent(), input="hello")
            print(f"Results from run(): {results}")
        except LimitExceededError as ex:
            print(
                f"Caught limit exceeded in solver: {ex.type}; value: {ex.value}; limit: {ex.limit}"
            )
            raise

        await generate(state)

        return state

    return solve


@agent
def my_agent():
    async def solve(state: AgentState) -> AgentState:
        # with working_time_limit(10):
        await get_model().generate("hi")
        await get_model().generate("hi")
        await get_model().generate("hi")

        return state

    return solve


@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="Just reply with Hello World",
                target="Hello World",
            )
        ],
        solver=[my_solver()],
        scorer=exact(),
        time_limit=4,
    )


# Why is there a difference between behaviour of time_limit and working_time_limit=4?
# In time_limit, no exception is raised from run(). anyio cancels the
# In working_time_limt, "our" code raises an exception.


if __name__ == "__main__":

    def repeat_forever(output: ModelOutput) -> Generator[ModelOutput, None, None]:
        while True:
            time.sleep(1.1)
            yield output

    output = ModelOutput.from_content("mockllm/model", "Hello World")
    output.usage = ModelUsage(total_tokens=1)
    model = get_model("mockllm/model", custom_outputs=repeat_forever(output))
    eval(hello_world(), model=model)
