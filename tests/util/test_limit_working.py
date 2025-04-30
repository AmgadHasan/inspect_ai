import asyncio

import pytest

from inspect_ai.util._limit import (
    LimitExceededError,
    check_working_time_limit,
    record_waiting_time,
    working_time_limit,
)


def test_can_record_waiting_time_with_no_active_limits() -> None:
    record_waiting_time(10)


def test_can_check_token_limit_with_no_active_limits() -> None:
    check_working_time_limit()


def test_validates_limit_parameter() -> None:
    with pytest.raises(ValueError):
        working_time_limit(-1)


async def test_can_create_with_none_limit() -> None:
    with working_time_limit(None):
        await asyncio.sleep(0.1)


def test_can_create_with_zero_limit() -> None:
    with working_time_limit(0):
        pass


def test_does_not_raise_error_when_limit_not_exceeded() -> None:
    with working_time_limit(10):
        check_working_time_limit()


async def test_raises_error_when_limit_exceeded() -> None:
    with working_time_limit(0.1):
        with pytest.raises(LimitExceededError) as exc_info:
            await asyncio.sleep(0.5)
            check_working_time_limit()

    assert exc_info.value.type == "working"
    assert 0.4 < exc_info.value.value < 0.6
    assert exc_info.value.limit == 0.1


async def test_raises_error_when_limit_repeatedly_exceeded() -> None:
    with working_time_limit(0.1):
        with pytest.raises(LimitExceededError):
            await asyncio.sleep(0.2)
            check_working_time_limit()
        with pytest.raises(LimitExceededError) as exc_info:
            await asyncio.sleep(0.1)
            check_working_time_limit()

    assert exc_info.value.type == "working"
    assert 0.2 < exc_info.value.value < 1
    assert exc_info.value.limit == 0.1


async def test_stack_can_trigger_outer_limit() -> None:
    with working_time_limit(0.1):
        with working_time_limit(10):
            with pytest.raises(LimitExceededError) as exc_info:
                await asyncio.sleep(0.2)
                check_working_time_limit()

    assert exc_info.value.limit == 0.1


async def test_stack_can_trigger_inner_limit() -> None:
    with working_time_limit(10):
        with working_time_limit(0.1):
            with pytest.raises(LimitExceededError) as exc_info:
                await asyncio.sleep(0.2)
                check_working_time_limit()

    assert exc_info.value.limit == 0.1


async def test_out_of_scope_limits_are_not_checked() -> None:
    with working_time_limit(0.1):
        pass

    await asyncio.sleep(0.2)
    check_working_time_limit()


async def test_outer_limit_is_checked_after_inner_limit_popped() -> None:
    with working_time_limit(0.1):
        with working_time_limit(10):
            pass

        with pytest.raises(LimitExceededError) as exc_info:
            await asyncio.sleep(0.2)
            check_working_time_limit()

    assert exc_info.value.limit == 0.1


async def test_subtracts_waiting_time() -> None:
    with working_time_limit(0.1):
        await asyncio.sleep(0.2)
        record_waiting_time(0.2)
        check_working_time_limit()

        await asyncio.sleep(0.4)
        with pytest.raises(LimitExceededError) as exc_info:
            check_working_time_limit()

    assert 0.3 < exc_info.value.value < 0.5
