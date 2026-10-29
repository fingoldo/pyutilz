"""Behaviour of ``pyutilz.performance.kernel_tuning.cache.cache_hooks``: ``LoggerHooks`` reports every hook at the level the
cache historically used (INFO for sweep start/end and invalidation, DEBUG for the rest), each message names the kernel and
the value that matters, and a custom implementation of the ``TuningHooks`` protocol is the documented extension point.
Nothing here touches hardware or disk."""

from __future__ import annotations

import logging

from pyutilz.performance.kernel_tuning.cache import cache_hooks
from pyutilz.performance.kernel_tuning.cache.cache_hooks import LoggerHooks


class TestLoggerHooksLevels:
    def test_sweep_start_and_end_are_info(self, caplog):
        with caplog.at_level(logging.INFO, logger=cache_hooks.logger.name):
            LoggerHooks().sweep_start("matmul", [1, 2])
            LoggerHooks().sweep_end("matmul", 3)
        levels = [rec.levelno for rec in caplog.records]
        assert levels == [logging.INFO, logging.INFO]

    def test_invalidation_is_info(self, caplog):
        with caplog.at_level(logging.INFO, logger=cache_hooks.logger.name):
            LoggerHooks().invalidation("matmul", "driver changed")
        assert [rec.levelno for rec in caplog.records] == [logging.INFO]

    def test_hits_misses_and_persists_are_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=cache_hooks.logger.name):
            hooks = LoggerHooks()
            hooks.cache_hit("matmul", {"n": 4}, {"block": 8})
            hooks.cache_miss("matmul", {"n": 4})
            hooks.persist("matmul", "x.json", 2)
            hooks.winner_chosen("matmul", {"block": 8}, "fastest")
            hooks.env_override("matmul", "naive")
            hooks.concurrent_sweep_detected("matmul")
        assert {rec.levelno for rec in caplog.records} == {logging.DEBUG}


class TestLoggerHooksMessages:
    def test_a_hit_names_the_kernel_and_the_dims_but_not_the_region(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=cache_hooks.logger.name):
            LoggerHooks().cache_hit("matmul", {"n": 4}, {"block": 8})
        message = caplog.records[0].getMessage()
        assert "matmul" in message and "{'n': 4}" in message
        assert "block" not in message

    def test_a_miss_names_the_kernel_and_dims(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=cache_hooks.logger.name):
            LoggerHooks().cache_miss("matmul", {"n": 4})
        assert "cache miss" in caplog.records[0].getMessage()

    def test_a_sweep_end_reports_the_region_count(self, caplog):
        with caplog.at_level(logging.INFO, logger=cache_hooks.logger.name):
            LoggerHooks().sweep_end("matmul", 3)
        assert "3 regions" in caplog.records[0].getMessage()

    def test_a_persist_reports_count_and_path(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=cache_hooks.logger.name):
            LoggerHooks().persist("matmul", "x.json", 2)
        message = caplog.records[0].getMessage()
        assert "2 regions" in message and "x.json" in message

    def test_an_invalidation_carries_its_reason(self, caplog):
        with caplog.at_level(logging.INFO, logger=cache_hooks.logger.name):
            LoggerHooks().invalidation("matmul", "driver changed")
        assert "driver changed" in caplog.records[0].getMessage()


class TestDefault:
    def test_the_module_default_is_a_logger_hooks_instance(self):
        assert isinstance(cache_hooks._DEFAULT_HOOKS, LoggerHooks)


class TestLoggedMessageText:
    @staticmethod
    def _one(caplog, call):
        with caplog.at_level(logging.DEBUG, logger=cache_hooks.logger.name):
            call()
        assert len(caplog.records) == 1
        return caplog.records[0].getMessage()

    def test_an_env_override_names_the_kernel_and_the_forced_choice(self, caplog):
        message = self._one(caplog, lambda: LoggerHooks().env_override("matmul", "naive"))
        assert message == "kernel_tuning_cache: matmul env override -> 'naive'"

    def test_a_miss_names_the_kernel_then_the_dims(self, caplog):
        message = self._one(caplog, lambda: LoggerHooks().cache_miss("matmul", {"n": 4}))
        assert message == "kernel_tuning_cache: matmul cache miss {'n': 4}"

    def test_a_sweep_start_names_the_kernel_and_its_axes(self, caplog):
        with caplog.at_level(logging.INFO, logger=cache_hooks.logger.name):
            LoggerHooks().sweep_start("matmul", ["n", "m"])
        assert caplog.records[0].getMessage() == "kernel_tuning_cache: matmul sweep starting (axes=['n', 'm'])"

    def test_a_winner_names_the_kernel_the_reason_and_the_region(self, caplog):
        message = self._one(caplog, lambda: LoggerHooks().winner_chosen("matmul", {"block": 8}, "fastest"))
        assert message == "kernel_tuning_cache: matmul winner (fastest): {'block': 8}"

    def test_a_self_healed_marker_names_the_kernel(self, caplog):
        message = self._one(caplog, lambda: LoggerHooks().concurrent_sweep_detected("matmul"))
        assert message == "kernel_tuning_cache: matmul reclaiming stale sweep marker (self-heal)"
