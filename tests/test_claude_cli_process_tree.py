"""The CLI child is killed as a tree, not as a PID.

``claude`` is a node process that can spawn children of its own, and on Windows ``Popen.kill()``
is TerminateProcess on one PID. A surviving grandchild keeps the inherited stdout and stderr
handles open, so the reader threads never reach EOF, their ``join(timeout=5)`` expires, and every
abandoned call leaves a daemon thread pinned to a pipe. Fails against the pre-fix tree, where the
unwind path called ``proc.kill()`` directly.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

import pyutilz.llm.claude_code_cli as cli


class _Proc:
    def __init__(self, alive=True, pid=4242):
        self.pid = pid
        self._alive = alive
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def kill(self):
        self.killed = True


@pytest.fixture
def taskkill_calls(monkeypatch):
    """Record what would have been spawned to kill the tree."""
    calls: "list[list[str]]" = []

    def _run(cmd, **_kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", _run)
    return calls


class TestTheWholeTreeIsEnded:
    @pytest.mark.skipif(sys.platform != "win32", reason="taskkill is the Windows branch")
    def test_windows_walks_the_tree_by_pid(self, taskkill_calls):
        proc = _Proc()

        cli._kill_process_tree(proc)

        assert taskkill_calls, "no tree kill was attempted; a grandchild would keep the pipes open"
        cmd = taskkill_calls[0]
        assert cmd[0] == "taskkill"
        assert "/T" in cmd, "without /T this kills the PID only, which is the bug"
        assert "/F" in cmd
        assert cmd[-1] == str(proc.pid)

    def test_the_direct_kill_still_runs_as_a_fallback(self, taskkill_calls):
        """Belt and braces: taskkill can fail, and POSIX may have no process group."""
        proc = _Proc()

        cli._kill_process_tree(proc)

        assert proc.killed is True

    def test_an_already_reaped_process_is_left_alone(self, taskkill_calls):
        proc = _Proc(alive=False)

        cli._kill_process_tree(proc)

        assert taskkill_calls == []
        assert proc.killed is False

    def test_a_failing_taskkill_does_not_raise(self, monkeypatch):
        """This runs on the unwind path, where the caller has a result that matters more."""

        def _boom(*_a, **_kw):
            raise OSError("taskkill missing")

        monkeypatch.setattr(cli.subprocess, "run", _boom)
        proc = _Proc()

        cli._kill_process_tree(proc)

        assert proc.killed is True

    def test_a_failing_direct_kill_still_leaves_the_tree_walked(self, taskkill_calls):
        """The fallback raising must not skip -- or undo -- the tree kill that ran before it."""

        class _Stubborn(_Proc):
            def kill(self):
                raise OSError("already gone")

        cli._kill_process_tree(_Stubborn(pid=777))

        assert taskkill_calls, "the tree kill was skipped"
        assert taskkill_calls[0][-1] == "777"


class TestTheChildGetsItsOwnGroupWhereThatIsMeaningful:
    def test_the_spawn_asks_for_a_new_session_off_windows(self, monkeypatch):
        """killpg needs a group to aim at; Windows has no equivalent and uses taskkill /T."""
        seen: "list[dict]" = []

        class _Pipe:
            def write(self, _s):
                pass

            def close(self):
                pass

            def __iter__(self):
                return iter(())

        class _Spawned(_Proc):
            stdin = _Pipe()
            stdout = _Pipe()
            stderr = _Pipe()
            args = ["claude"]

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def wait(self, timeout=None):
                return 0

        def _popen(cmd, **kwargs):
            seen.append(kwargs)
            return _Spawned()

        monkeypatch.setattr(cli.subprocess, "Popen", _popen)
        monkeypatch.setattr(cli, "_consume_cli_stream", lambda *_a, **_kw: ("ok", None, False, None))
        monkeypatch.setattr(cli, "_kill_process_tree", lambda _p: None)

        cli.run_cli(["claude"], "hi", {}, ".", 5.0, [], threading.Event())

        assert seen[0]["start_new_session"] is (sys.platform != "win32")
