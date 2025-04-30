import asyncio

import pytest

from inspect_ai.util._limit import LimitExceededError, time_limit


def test_validates_limit_parameter() -> None:
    with pytest.raises(ValueError):
        time_limit(-0.1)


async def test_can_create_with_none_limit() -> None:
    with time_limit(None):
        pass


async def test_can_create_with_zero_limit() -> None:
    with pytest.raises(LimitExceededError):
        with time_limit(0):
            await asyncio.sleep(0.1)


async def test_does_not_raise_error_when_limit_not_exceeded() -> None:
    with time_limit(10):
        pass


async def test_raises_error_when_limit_exceeded() -> None:
    with pytest.raises(LimitExceededError) as exc_info:
        with time_limit(0.1):
            await asyncio.sleep(0.5)

    assert exc_info.value.type == "time"
    assert exc_info.value.value == 0.1
    assert exc_info.value.limit == 0.1


async def test_out_of_scope_limits_are_not_checked() -> None:
    with time_limit(0.1):
        pass

    await asyncio.sleep(0.5)


async def test_outer_limits_are_enforced() -> None:
    with pytest.raises(LimitExceededError) as exc_info:
        with time_limit(0.1):
            with time_limit(10):
                await asyncio.sleep(1)

    assert exc_info.value.value == 0.1


async def test_inner_limits_are_enforced() -> None:
    with pytest.raises(LimitExceededError) as exc_info:
        with time_limit(10):
            with time_limit(0.1):
                await asyncio.sleep(1)

    assert exc_info.value.value == 0.1


async def test_enter_context_manager_multiple_times() -> None:
    limit = time_limit(1)

    with limit:
        pass
    with pytest.raises(RuntimeError) as exc_info:
        with limit:
            pass

    assert "Cannot enter a time limit context manager instance multiple times" in str(
        exc_info.value
    )


async def test_enter_context_manager_multiple_times_nested() -> None:
    limit = time_limit(1)

    with pytest.raises(RuntimeError) as exc_info:
        with limit:
            with limit:
                pass

    assert "Cannot enter a time limit context manager instance multiple times" in str(
        exc_info.value
    )
