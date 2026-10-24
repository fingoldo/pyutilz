"""2026-09-03 audit F10: concurrent first touch of a PerCallAttr on one shared instance never loses a write.

Split out of ``test_domain_db_web_cloud_llm_text_audit_20260903_llm.py`` for the 1,000-line test-module budget.
"""

from __future__ import annotations

import sys
import threading

from pyutilz.llm.base import PerCallAttr


class TestF10PerCallAttrThreadSafety:
    def test_concurrent_first_touch_never_loses_a_write(self):
        class _Holder:
            value: PerCallAttr = PerCallAttr(lambda: "DEFAULT")

        # 16 threads are started ONCE and reused for all 200 rounds. Starting 3200 threads cost ~300 s on
        # 3.14/Windows (thread creation alone measured ~85 ms per thread here, independent of PerCallAttr). Each
        # round still uses a fresh instance, so every round is a first touch, and all 16 workers are released
        # together from a barrier with the 1 ns switch interval in force, as before.
        rounds, threads_n = 200, 16
        holders = [_Holder() for _ in range(rounds)]
        results: list[list[str]] = [[""] * threads_n for _ in range(rounds)]
        start = threading.Barrier(threads_n + 1)
        done = threading.Barrier(threads_n + 1)

        def worker(index: int) -> None:
            try:
                for r in range(rounds):
                    start.wait()
                    holders[r].value = "thread-%d" % index
                    results[r][index] = holders[r].value
                    done.wait()
            except threading.BrokenBarrierError:  # the main thread aborted after a failed round
                return

        switch_interval = sys.getswitchinterval()
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads_n)]
        for t in threads:
            t.start()
        try:
            for r in range(rounds):
                sys.setswitchinterval(1e-9)
                start.wait(timeout=60)
                done.wait(timeout=60)
                sys.setswitchinterval(switch_interval)
                assert results[r] == ["thread-%d" % i for i in range(threads_n)]
        except BaseException:
            start.abort()
            done.abort()
            raise
        finally:
            sys.setswitchinterval(switch_interval)
            for t in threads:
                t.join(timeout=10)
