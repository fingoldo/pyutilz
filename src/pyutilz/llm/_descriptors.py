"""Descriptors that keep a shared, factory-cached provider safe across tasks, threads and event loops.

``PerCallAttr`` (per-call state isolated per asyncio Task), ``LazySemaphore`` (one concurrency semaphore per
event loop) and ``PerLoopHTTPClient`` (one ``httpx.AsyncClient`` per event loop). Split out of ``base.py`` for
the 1,000-line module limit, moved verbatim (``pyutilz.dev.freevar_analysis.split_out_module``); ``base``
re-exports every name, so ``from pyutilz.llm.base import PerCallAttr`` keeps working.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
import weakref
from typing import Any, Callable


class PerCallAttr:
    """Descriptor for a provider's "last successful call" state (e.g. ``last_tool_calls``,
    ``_last_usage``), backed by a private ``contextvars.ContextVar`` per (instance, attribute).

    Regression fix (2026-07-21 audit round 2, HIGH): these used to be plain instance attributes,
    written unconditionally at the end of every ``generate()`` call. ``generate_batch()`` fires
    N concurrent ``self.generate()`` calls on ONE shared, cached provider instance (see
    ``llm.factory.get_llm_provider``'s whole reason for existing) via ``asyncio.create_task`` --
    a write from one in-flight task was visible to every other task reading the same plain
    attribute, so a caller reading e.g. ``provider.last_tool_calls`` right after a batch item is
    yielded could silently get a DIFFERENT request's data (verified with a standalone repro:
    ``generate_batch()`` yielded ``id='req-0'`` while ``provider.last_tool_calls`` already
    reflected ``id='req-2'``).

    ``asyncio.create_task()`` gives each Task its own COPY of the current context, so a
    ``contextvars.ContextVar`` set inside one task is invisible to every other task -- this
    closes the cross-task race. A direct (non-batched) ``await provider.generate(...)`` keeps
    working exactly as before: no task boundary is crossed, so the write and the caller's
    immediately-following read share the same context.

    Does NOT fix (by design -- this is the correct, intentional behavior): the outer caller of
    ``generate_batch()`` runs in yet another context than any individual request task, so it can
    no longer read a completed batch item's metadata off the provider instance at all -- it must
    come from the yielded dict instead (see ``generate_batch``'s ``usage``/``tool_calls``/etc.
    keys), which is exactly the fix the audit's own report recommended.
    """

    def __init__(self, default_factory: Callable[[], Any]) -> None:
        self._default_factory = default_factory
        self._name = "_unnamed"
        # One ContextVar per DESCRIPTOR (i.e. per class attribute), created here rather than
        # lazily per (instance, attribute) on first touch. Two independent bugs are closed by
        # that single change:
        #  * the lazy per-instance creation was a check-then-act with no lock, so two threads
        #    first touching the same attribute on one shared (factory-cached) provider could each
        #    build a DISTINCT ContextVar; the loser wrote to a var no longer reachable and its own
        #    next read fell through to the default -- indistinguishable from "the model returned
        #    nothing" (2026-09-03 audit F10, reproduced in 287 of 3000 threaded trials).
        #  * CPython documents ContextVars as objects to create at module/class scope and never
        #    inside a function, because live Context objects keep strong references to them. One
        #    var per instance meant a service constructing a provider per request (the
        #    unhashable-kwargs path in llm.factory bypasses the LRU) minted vars nothing could
        #    reclaim (F41). The count is now bounded by the number of declared class attributes.
        # The var's VALUE is a per-instance mapping keyed by id(), validated through a weakref so
        # a recycled id can never surface a dead instance's state. It is replaced copy-on-write on
        # every set, never mutated in place, so a value set inside an asyncio Task stays invisible
        # to every other Task -- the whole point of this descriptor.
        self._var: contextvars.ContextVar[dict[int, Any]] = contextvars.ContextVar("PerCallAttr._unnamed")

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._var = contextvars.ContextVar(f"{owner.__name__}.{name}")

    def default(self) -> Any:
        """Return a freshly built default value for this attribute (used by ``_reset_per_call_state``)."""
        return self._default_factory()

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        entry = self._var.get({}).get(id(instance))
        if entry is not None:
            ref, value = entry
            if ref() is instance:
                return value
        return self._default_factory()

    def __set__(self, instance: Any, value: Any) -> None:
        try:
            ref = weakref.ref(instance)
        except TypeError:
            # A strong-reference fallback used to live here. It pinned the instance, and its last value, in the
            # context's store for as long as the context lived: the per-request leak this descriptor exists to
            # avoid. A type that declares __slots__ must list "__weakref__" to carry per-call state.
            raise TypeError(f"PerCallAttr {self._name!r} needs weak references to {type(instance).__name__}; add '__weakref__' to its __slots__") from None
        # The copy that copy-on-write needs anyway is also where dead instances are dropped. Without the filter every
        # short-lived provider left its last usage/tool calls/citations behind for the life of the context, and each
        # set copied all of them, so N short-lived instances cost O(N^2) in total.
        store = {key: entry for key, entry in self._var.get({}).items() if entry[0]() is not None}
        store[id(instance)] = (ref, value)
        self._var.set(store)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """The running event loop, or None outside one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class _NoLoop:
    """Weak-referenceable stand-in key for "no loop is running"."""


_NO_LOOP = _NoLoop()

#: Set while shutting providers down (``llm.factory``'s atexit handler). ``PerLoopHTTPClient`` then returns the
#: client it already has instead of cloning one for the shutdown loop only to close the clone.
_NO_REBIND: contextvars.ContextVar[bool] = contextvars.ContextVar("pyutilz_llm_no_client_rebind", default=False)


class LazySemaphore:
    """Descriptor for a provider's concurrency-limiting ``asyncio.Semaphore``: one per event loop, built lazily.

    Lazy because on Python 3.9 ``asyncio.Semaphore.__init__`` binds ``asyncio.get_event_loop()`` eagerly, and a
    provider is normally constructed with no loop running (``llm.factory.get_llm_provider``, a sync test, import
    time), which raised ``RuntimeError: There is no current event loop`` on 3.9 only.

    Per LOOP because the factory caches providers process-wide while a script, notebook or test suite may call
    ``asyncio.run`` many times. A single semaphore bound to the first loop raised ``RuntimeError: ... is bound to a
    different event loop`` on the second one, and only under contention, so it looked flaky. The per-loop map is
    weak, so a finished loop's semaphore goes with it.

    An explicitly assigned semaphore (``provider.semaphore = asyncio.Semaphore(1)``) is the caller's choice and is
    returned as-is on every loop.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._per_loop_key = f"_{name}_per_loop"

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        explicit = instance.__dict__.get(self._name)
        if explicit is not None:
            return explicit
        per_loop: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] | None = instance.__dict__.get(self._per_loop_key)
        if per_loop is None:
            per_loop = weakref.WeakKeyDictionary()
            instance.__dict__[self._per_loop_key] = per_loop
        loop = _running_loop()
        key: Any = loop if loop is not None else _NO_LOOP
        value = per_loop.get(key)
        if value is None:
            value = asyncio.Semaphore(instance._max_concurrent)
            # Stored through the instance so every later __get__ on this loop returns this same object.
            instance.__dict__[self._per_loop_key][key] = value
        return value

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self._name] = value


def _is_async_httpx_client(value: Any) -> bool:
    """True for an ``httpx.AsyncClient``, without importing httpx into a process that never did."""
    httpx = sys.modules.get("httpx")
    return httpx is not None and isinstance(value, httpx.AsyncClient)


def _clone_async_client(client: Any) -> Any:
    """A fresh ``httpx.AsyncClient`` configured like ``client``, whose connection pool no loop owns yet.

    A non-network transport (a test's ``MockTransport``) holds no sockets and is shared. A network transport is
    rebuilt from its pool settings, read with ``getattr`` defaults because they are httpcore internals.
    """
    import httpx

    transport = getattr(client, "_transport", None)
    new_transport: Any
    if transport is not None and not isinstance(transport, httpx.AsyncHTTPTransport):
        new_transport = transport
    else:
        pool = getattr(transport, "_pool", None)
        kwargs: dict[str, Any] = {}
        ssl_context = getattr(pool, "_ssl_context", None)
        if ssl_context is not None:
            kwargs["verify"] = ssl_context
        for attr, name in (("_http1", "http1"), ("_http2", "http2")):
            flag = getattr(pool, attr, None)
            if isinstance(flag, bool):
                kwargs[name] = flag
        if pool is not None and hasattr(pool, "_max_connections"):
            kwargs["limits"] = httpx.Limits(
                max_connections=getattr(pool, "_max_connections", None),
                max_keepalive_connections=getattr(pool, "_max_keepalive_connections", None),
                keepalive_expiry=getattr(pool, "_keepalive_expiry", 5.0),
            )
        retries = getattr(pool, "_retries", None)
        if isinstance(retries, int):
            kwargs["retries"] = retries
        new_transport = httpx.AsyncHTTPTransport(**kwargs)
    return httpx.AsyncClient(
        base_url=client.base_url,
        headers=client.headers,
        params=client.params,
        cookies=client.cookies,
        auth=client.auth,
        timeout=client.timeout,
        follow_redirects=client.follow_redirects,
        max_redirects=client.max_redirects,
        event_hooks=client.event_hooks,
        trust_env=client.trust_env,
        transport=new_transport,
    )


class PerLoopHTTPClient:
    """Descriptor that gives each event loop its own copy of a provider's ``httpx.AsyncClient``.

    The factory caches providers process-wide, and an ``httpx.AsyncClient``'s pooled sockets belong to the loop
    that opened them: after one ``asyncio.run`` returned, the next reused sockets of a closed loop. The client
    assigned in ``__init__`` serves the first loop that uses it; every other loop gets a clone with the same
    configuration (``_clone_async_client``), held weakly by loop. Anything that is not an ``httpx.AsyncClient``
    (an SDK client, a test double) is returned unchanged, and so is any client outside a running loop.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._per_loop_key = f"_{name}_per_loop"
        self._home_key = f"_{name}_home_loop"

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        try:
            original = instance.__dict__[self._name]
        except KeyError:
            raise AttributeError(self._name) from None
        if not _is_async_httpx_client(original):
            return original
        loop = _running_loop()
        if loop is None or _NO_REBIND.get():
            return original
        home = instance.__dict__.get(self._home_key)
        if home is None:
            instance.__dict__[self._home_key] = weakref.ref(loop)
            return original
        if home() is loop:
            return original
        per_loop: weakref.WeakKeyDictionary[Any, Any] = instance.__dict__.setdefault(self._per_loop_key, weakref.WeakKeyDictionary())
        clone = per_loop.get(loop)
        if clone is None:
            clone = _clone_async_client(original)
            per_loop[loop] = clone
        return clone

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self._name] = value
        instance.__dict__.pop(self._home_key, None)
        instance.__dict__.pop(self._per_loop_key, None)

    def all_clients(self, instance: Any) -> list[Any]:
        """Every client ``instance`` holds: the assigned one, then each live per-loop clone."""
        out: list[Any] = []
        if self._name in instance.__dict__:
            out.append(instance.__dict__[self._name])
        per_loop = instance.__dict__.get(self._per_loop_key)
        if per_loop is not None:
            out.extend(per_loop.values())
        return out

    def home_loop(self, instance: Any) -> asyncio.AbstractEventLoop | None:
        """The loop that first used the assigned client, while that loop object is alive."""
        home = instance.__dict__.get(self._home_key)
        return home() if home is not None else None

    def home_loop_gone(self, instance: Any) -> bool:
        """True when the assigned client was used on a loop that is now closed or garbage-collected."""
        home = instance.__dict__.get(self._home_key)
        if home is None:
            return False
        loop = home()
        return loop is None or loop.is_closed()
