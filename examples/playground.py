import asyncio
import time

import anyio

from inspect_ai.model._model_output import ModelUsage
from inspect_ai.util._limit import check_token_limit, record_model_usage, token_limit


async def main():
    try:
        with anyio.fail_after(1) as cancel_scope:
            try:
                await asyncio.sleep(2)
                # time.sleep(2)
            except Exception as ex:
                print(f"Time inner caught exception: {ex}")
    except Exception as ex:
        print(f"Time outer caught exception: {ex}")

    try:
        with token_limit(1):
            try:
                record_model_usage(ModelUsage(total_tokens=2))
                check_token_limit()
            except Exception as ex:
                print(f"Token inner caught exception: {ex}")
    except Exception as ex:
        print(f"Token outer caught exception: {ex}")


if __name__ == "__main__":
    asyncio.run(main())
