"""sweep_backend_grid skips the cells a variant already dominates on the way up the axes.

Costs are synthetic (the ranking step is replaced by a cost model) so the tests pin the pruning rule, not timing noise.
"""

from __future__ import annotations

from pyutilz.dev import benchmarking as bm


def _sweep(monkeypatch, cost, axes, **kw):
    """Run the sweep with ``cost(name, dims) -> ms`` standing in for the timed ranking; returns (regions, cells whose inputs were built)."""
    built: list = []

    def make_inputs(dims):
        built.append(dict(dims))
        return (0,)

    def fake_rank(candidates, repeats, synchronize_gpu, ranking="robust"):
        dims = built[-1]
        return {name: cost(name, dims) for name in candidates}

    monkeypatch.setattr(bm, "_rank_candidates", fake_rank)
    variants = {"cpu": lambda *a: 0.0, "gpu": lambda *a: 0.0}
    regions = bm.sweep_backend_grid(variants, axes, make_inputs, reference="cpu", repeats=1, **kw)
    return regions, built


def _by_cell(regions, dims):
    """Winner recorded for the cell bounded by ``dims`` (axis values)."""
    for r in regions:
        if all(r.get(f"{d}_max") == v for d, v in dims.items()):
            return r["backend_choice"]
    raise AssertionError(f"no region for {dims}")


def test_a_variant_that_dominates_its_smaller_neighbours_ends_the_measuring(monkeypatch):
    """GPU wins by 10x with a rival already costing seconds: the larger cells are not built or timed, but still get GPU as their decision."""
    axes = {"n": [1, 2, 3, 4, 5]}
    regions, built = _sweep(monkeypatch, lambda name, d: 10.0 * d["n"] * (1.0 if name == "gpu" else 1000.0), axes)
    assert [b["n"] for b in built] == [1, 2]  # cell 1 has no neighbour; cell 2 is checked against it and still measured only if cell 1 did not settle it
    assert all(_by_cell(regions, {"n": n}) == "gpu" for n in axes["n"])
    assert next(r for r in regions if r.get("n_max") is None)["backend_choice"] == "gpu"  # catch-all keeps the largest cell's winner


def test_cheap_cells_are_always_measured(monkeypatch):
    """A rival costing less than prune_min_ms is never skipped: measuring it is cheap and the answer might change."""
    axes = {"n": [1, 2, 3, 4]}
    _, built = _sweep(monkeypatch, lambda name, d: 1.0 if name == "gpu" else 50.0, axes)
    assert [b["n"] for b in built] == [1, 2, 3, 4]


def test_a_narrow_win_is_not_pruned(monkeypatch):
    """A win under prune_ratio is not 'dominating': the crossover may still move, so every cell is timed."""
    axes = {"n": [1, 2, 3, 4]}
    _, built = _sweep(monkeypatch, lambda name, d: 5000.0 * d["n"] if name == "cpu" else 3000.0 * d["n"], axes)
    assert [b["n"] for b in built] == [1, 2, 3, 4]


def test_a_crossover_is_still_found(monkeypatch):
    """CPU wins small cells, GPU wins big ones (by a wide margin): the winner flips and the flip is measured, not pruned past."""
    axes = {"n": [1, 2, 3, 4, 5, 6]}

    def cost(name, d):
        """CPU linear, GPU flat: they cross between n=3 and n=4."""
        return 1500.0 * d["n"] if name == "cpu" else 5000.0

    regions, _ = _sweep(monkeypatch, cost, axes)
    assert [_by_cell(regions, {"n": n}) for n in axes["n"]] == ["cpu", "cpu", "cpu", "gpu", "gpu", "gpu"]


def test_two_axes_prune_only_where_every_direction_is_settled(monkeypatch):
    """With the GPU dominating the whole 4x4 grid, the 2x2 corner is measured to establish the trend on each axis and the cells beyond it are skipped."""
    axes = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
    regions, built = _sweep(monkeypatch, lambda name, d: 2000.0 if name == "gpu" else 200000.0, axes)
    measured = {(b["a"], b["b"]) for b in built}
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= measured  # two points per axis are needed to see a trend
    assert (4, 4) not in measured and (3, 3) not in measured
    assert all(_by_cell(regions, {"a": a, "b": b}) == "gpu" for a in axes["a"] for b in axes["b"])


def test_an_undecided_line_keeps_the_cells_after_it_measured(monkeypatch):
    """Along b=1 the CPU only ties; the cells that would extend that line's trend are measured, never assumed."""
    axes = {"a": [1, 2, 3], "b": [1, 2, 3]}
    _, built = _sweep(monkeypatch, lambda name, d: 2000.0 if name == "gpu" or d["b"] == 1 else 200000.0, axes)
    assert (1, 3) in {(b["a"], b["b"]) for b in built}


def test_pruning_can_be_turned_off(monkeypatch):
    """prune_dominated=False measures every cell."""
    axes = {"n": [1, 2, 3, 4]}
    _, built = _sweep(monkeypatch, lambda name, d: 10.0 if name == "gpu" else 100000.0, axes, prune_dominated=False)
    assert [b["n"] for b in built] == [1, 2, 3, 4]


def test_skipped_cells_do_not_run_any_variant(monkeypatch):
    """Skipping saves the expensive calls themselves: no variant is invoked at a skipped cell."""
    calls = []
    monkeypatch.setattr(bm, "_rank_candidates", lambda candidates, repeats, synchronize_gpu, ranking="robust": {n: (10.0 if n == "gpu" else 100000.0) for n in candidates})
    variants = {"cpu": lambda *a: calls.append(("cpu", a)) or 0.0, "gpu": lambda *a: calls.append(("gpu", a)) or 0.0}
    bm.sweep_backend_grid(variants, {"n": [1, 2, 3, 4, 5]}, lambda dims: (dims["n"],), reference="cpu", repeats=1)
    assert {a[0] for _, a in calls} == {1, 2}


def test_prune_ratio_is_honoured(monkeypatch):
    """A 1.5x lead is not dominating by default (every cell measured) but is with prune_ratio=1.2."""
    axes = {"n": [1, 2, 3, 4]}

    def cost(name, d):
        """GPU 1.5x faster than the CPU everywhere, both costly."""
        return 2000.0 if name == "gpu" else 3000.0

    _, strict = _sweep(monkeypatch, cost, axes)
    _, loose = _sweep(monkeypatch, cost, axes, prune_ratio=1.2)
    assert [b["n"] for b in strict] == [1, 2, 3, 4]
    assert [b["n"] for b in loose] == [1, 2]


def test_a_shrinking_lead_is_not_pruned(monkeypatch):
    """The winner still wins by a wide margin but the margin is closing step by step: the cells are measured to see where it ends."""
    axes = {"n": [1, 2, 3, 4]}
    _, built = _sweep(monkeypatch, lambda name, d: 1000.0 if name == "gpu" else 20000.0 / d["n"], axes)
    assert [b["n"] for b in built] == [1, 2, 3, 4]
