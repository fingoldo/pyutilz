def test_cli_stream_consumer_is_a_separate_function():
    """`_generate_cli` was split at 25 complexity; the ratchet only turns down."""
    import queue

    from pyutilz.llm.claude_code_provider import _consume_cli_stream

    q: "queue.Queue" = queue.Queue()
    q.put('{"type": "system", "subtype": "init"}')
    q.put('{"type": "result", "subtype": "success", "result": "hello"}')
    assert _consume_cli_stream(q, timeout=5.0)[:3] == ("hello", None, False)

    q2: "queue.Queue" = queue.Queue()
    q2.put('{"type": "result", "subtype": "error", "error": "nope"}')
    assert _consume_cli_stream(q2, timeout=5.0)[:3] == (None, "nope", False)

    q3: "queue.Queue" = queue.Queue()
    q3.put(None)  # reader EOF sentinel: neither a result nor a timeout
    assert _consume_cli_stream(q3, timeout=5.0)[:3] == (None, None, False)
