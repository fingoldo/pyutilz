import ast


def test_f10_end_line_survives_a_node_without_lineno():
    """`ast.AST` genuinely has no `lineno`; the old fallback reached for it unguarded, mid-rewrite."""
    from pyutilz.dev.freevar_analysis import _end_line

    assert _end_line(ast.Load()) == 0
    assert _end_line(ast.parse("x = 1").body[0]) == 1
