"""py-ci-shared gates adopted with the e13229e pin, all at zero findings: concurrency/IO hygiene, numerics, CI wiring, test hygiene.

Floors sit well under the measured counts (src/pyutilz about 270 parsed files, tests/ about 320). Not wired, with the reason:
``stdlib_json_ban`` (openrouter_speech hashes request bodies with stdlib json ``sort_keys``; the ban is an opt-in house style),
``pickle_state_completeness`` (flags the lock-guarded counters of SpeechAccounting that are meant to be pickled),
``env_flag_parsing`` (flags path and number variables, not boolean flags), ``runtime_registry_mutation`` (``kernel_tuner`` is a
decorator that registers at import, so every process that imports the kernel module replays it), ``drifted_duplicate_literals``
(the 15.0 s and 120.0 s defaults sit in unrelated packages that must not import each other), ``workflow_runner_labels``
(pinning runner images changes the CI matrix and cannot be verified here), ``patch_target_read_at_call`` (38 inert patches
in tests), ``wallclock_test_waits`` (35 sleeping tests), ``disposition_test_references`` (46 audit rows name tests that were
renamed); the last three are open work, see audits/2026-11-05/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from py_ci_shared.api_floor import assert_api_floor
from py_ci_shared.audit_wave_filenames import assert_no_new_audit_wave_filenames
from py_ci_shared.baseline_hygiene import assert_baseline_is_honest
from py_ci_shared.cancellation_prone_moments import assert_cancellation_prone_moments
from py_ci_shared.ci_default_branch_never_cancelled import assert_ci_default_branch_never_cancelled
from py_ci_shared.ci_install_covers_conftest import assert_ci_install_covers_conftest
from py_ci_shared.ci_install_covers_entry_imports import assert_ci_install_covers_entry_imports
from py_ci_shared.ci_pin_version_skew import assert_installed_matches_ci_pin
from py_ci_shared.ci_workflow_paths import assert_workflow_paths_exist
from py_ci_shared.clock_day_boundary import assert_no_clock_day_boundary
from py_ci_shared.commit_metadata import assert_commit_metadata
from py_ci_shared.comment_repo_paths import assert_comment_repo_paths
from py_ci_shared.committed_line_endings import assert_committed_line_endings
from py_ci_shared.connect_error_echo import assert_no_connect_error_echo
from py_ci_shared.connection_liveness_kwargs import assert_every_connection_has_liveness_kwargs
from py_ci_shared.constant_fallback_cache_key import assert_constant_fallback_cache_key
from py_ci_shared.coverage_config_parity import assert_coverage_config_parity
from py_ci_shared.cuda_kernel_integer_width import assert_cuda_kernel_integer_width
from py_ci_shared.discarded_model_copy import assert_no_discarded_model_copy
from py_ci_shared.env_write_restore import assert_env_write_restore
from py_ci_shared.git_child_env import assert_no_new_unscrubbed_git_spawns
from py_ci_shared.hardcoded_seed_in_library import assert_hardcoded_seed_in_library
from py_ci_shared.hardcoded_token_ceilings import assert_no_hardcoded_token_ceilings
from py_ci_shared.id_keyed_cache_validates_identity import assert_id_keyed_cache_validates_identity
from py_ci_shared.import_time_global_side_effects import assert_import_time_global_side_effects
from py_ci_shared.lf_file_writes import assert_no_crlf_writes
from py_ci_shared.machine_specific_paths import assert_no_machine_specific_paths
from py_ci_shared.module_state_test_reset import assert_module_state_test_reset
from py_ci_shared.optional_imports_guarded import assert_optional_imports_guarded
from py_ci_shared.order_losing_filters import assert_no_order_losing_filters
from py_ci_shared.package_doctests import assert_package_doctests_pass
from py_ci_shared.persisted_negative_probe import assert_persisted_negative_probe
from py_ci_shared.plotly_annotation_loop import assert_no_plotly_annotation_loops
from py_ci_shared.polars_null_equality import assert_polars_null_equality
from py_ci_shared.predictable_tmp_path import assert_predictable_tmp_path
from py_ci_shared.reiterated_iterable_params import assert_no_reiterated_iterable_params
from py_ci_shared.rollback_then_continue import assert_no_rollback_then_continue
from py_ci_shared.secret_assertion_operands import assert_no_secret_assertion_operands
from py_ci_shared.sibling_floor_skew import assert_sibling_floor_skew
from py_ci_shared.single_shot_timing_assertion import assert_single_shot_timing_assertion
from py_ci_shared.skip_as_pass import assert_no_new_skip_as_pass
from py_ci_shared.standard_stream_restore import assert_standard_streams_restored_only_if_still_ours
from py_ci_shared.stub_signature_parity import assert_stub_signature_parity
from py_ci_shared.survivorship_scoring import assert_no_survivorship_scoring
from py_ci_shared.timezone_honest import assert_timezone_honest
from py_ci_shared.unread_function_params import assert_unread_function_params
from py_ci_shared.unread_init_params import assert_no_unread_init_params
from py_ci_shared.unresolved_module_attributes import assert_no_unresolved_module_attributes
from py_ci_shared.unsafe_deserialization import assert_unsafe_deserialization
from py_ci_shared.vacuous_loop_assertions import assert_no_new_floorless_loop
from py_ci_shared.validation_errors_echo_input import assert_validation_errors_echo_input
from py_ci_shared.vendored_internal_imports import assert_no_vendored_internal_imports
from py_ci_shared.wrapper_protocol_parity import assert_wrapper_protocol_parity

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "pyutilz"
TESTS = REPO_ROOT / "tests"
META = Path(__file__).resolve().parent
VACUOUS_LOOP_BASELINE = META / "_vacuous_loop_baseline.json"
FLOOR = 150


def _py_files(root: Path) -> list[Path]:
    """Every Python file under *root*, sorted so a finding's order is stable across runs."""
    return sorted(root.rglob("*.py"))


# Source-tree gates that take the package root.
_SRC_GATES = [
    assert_cancellation_prone_moments,
    assert_no_connect_error_echo,
    assert_every_connection_has_liveness_kwargs,
    assert_constant_fallback_cache_key,
    assert_cuda_kernel_integer_width,
    assert_hardcoded_seed_in_library,
    assert_no_hardcoded_token_ceilings,
    assert_id_keyed_cache_validates_identity,
    assert_import_time_global_side_effects,
    assert_no_crlf_writes,
    assert_no_machine_specific_paths,
    assert_persisted_negative_probe,
    assert_no_plotly_annotation_loops,
    assert_polars_null_equality,
    assert_predictable_tmp_path,
    assert_no_reiterated_iterable_params,
    assert_no_rollback_then_continue,
    assert_timezone_honest,
    assert_unread_function_params,
    assert_unsafe_deserialization,
    assert_validation_errors_echo_input,
    assert_no_vendored_internal_imports,
    assert_wrapper_protocol_parity,
]


@pytest.mark.parametrize("gate", _SRC_GATES, ids=lambda g: g.__module__.rsplit(".", 1)[-1])
def test_source_tree_gate_is_clean(gate):
    """Each shared source-tree gate finds nothing under src/pyutilz (its own module docstring says what it looks for)."""
    gate(SRC, min_files=FLOOR)


def test_no_unresolved_attribute_on_a_first_party_module():
    """`m.NAME` on an imported first-party module must name something the module defines."""
    assert_no_unresolved_module_attributes([SRC], min_files=FLOOR)


def test_no_unread_constructor_parameters():
    """A constructor parameter nothing reads is a knob that does nothing."""
    assert_no_unread_init_params(_py_files(SRC), REPO_ROOT, min_files=FLOOR)


def test_no_row_selection_that_loses_order():
    """A mask built from an index returns rows in index order, not in the order the caller asked for."""
    assert_no_order_losing_filters(_py_files(SRC), REPO_ROOT, min_files=FLOOR)


def test_no_survivorship_scoring():
    """A metric computed only over the rows whose prediction happened to be finite hides the failures."""
    assert_no_survivorship_scoring(_py_files(SRC), REPO_ROOT, min_files=FLOOR)


def test_no_discarded_model_copy():
    """`cfg = cfg.model_copy(update=...)` into a local that reaches nothing is a no-op."""
    assert_no_discarded_model_copy(_py_files(SRC), REPO_ROOT, {}, min_files=FLOOR)


def test_module_state_written_by_production_code_is_reset_by_a_test():
    """Module-level mutable state a function writes needs a test-side reset, or tests order-depend on each other."""
    assert_module_state_test_reset(SRC, TESTS, min_files=FLOOR)


def test_standard_streams_are_restored_only_if_still_ours():
    """Swapping sys.stdout/sys.stderr and restoring blindly can restore a closed stream."""
    assert_standard_streams_restored_only_if_still_ours([SRC, TESTS], min_files=FLOOR)


def test_optional_extra_imports_are_guarded():
    """A module-level import of a package that is only an optional extra fails for users who did not install it."""
    assert_optional_imports_guarded(REPO_ROOT)


def test_no_api_newer_than_the_python_floor():
    """requires-python is 3.8; stdlib API added later must sit behind a version or hasattr guard."""
    assert_api_floor(REPO_ROOT, min_files=FLOOR)


def test_package_doctests_run():
    """The doctests the package ships actually run, and there are some to run."""
    assert_package_doctests_pass("pyutilz")


# Test-tree gates.


def test_no_test_reads_the_clock_near_a_day_boundary():
    """A test that shifts the real clock by part of a day fails for part of every day."""
    assert_no_clock_day_boundary(TESTS, min_files=250)


def test_environment_writes_in_tests_are_restored():
    """An os.environ write inside a test function leaks into every later test in the process."""
    assert_env_write_restore(TESTS, min_files=250)


def test_no_single_shot_timing_assertion():
    """A wall-clock ratio or ceiling from one measurement per side is flaky on a loaded runner."""
    assert_single_shot_timing_assertion(TESTS, min_files=250)


def test_monkeypatched_stubs_accept_every_parameter_of_what_they_replace():
    """A stub with a narrower signature passes until the code under test calls it with the new argument."""
    assert_stub_signature_parity(TESTS, min_files=250)


def test_test_files_are_not_named_after_an_audit_wave():
    """A test file is named after what it covers, not after the audit round that produced it."""
    assert_no_new_audit_wave_filenames(TESTS, min_files=250)


def test_assertions_do_not_print_secrets():
    """pytest prints every operand of a failed assert; an environment value or DSN must be folded to a bool first."""
    assert_no_secret_assertion_operands(_py_files(TESTS), min_files=250)


def test_no_loop_of_conditional_asserts_that_can_iterate_zero_times():
    """A for loop whose body only asserts conditionally passes when the collection is empty."""
    assert_no_new_floorless_loop(_py_files(TESTS), REPO_ROOT, VACUOUS_LOOP_BASELINE, min_files=250)


def test_vacuous_loop_baseline_is_empty():
    """The baseline of accepted floorless loops stays empty: new ones are fixed, not accepted."""
    assert json.loads(VACUOUS_LOOP_BASELINE.read_text(encoding="utf-8")) == {}


# Repository and CI wiring.


def test_installed_py_ci_shared_matches_the_ci_pin():
    """The version the workflows pin (the trailing `# vX.Y.Z`) is the version this interpreter runs."""
    assert_installed_matches_ci_pin(REPO_ROOT, min_files=1)


def test_ci_install_covers_the_imports_of_the_entry_points():
    """The extras a CI job installs must cover the module-level third-party imports of what it runs."""
    assert_ci_install_covers_entry_imports(REPO_ROOT, min_files=1)


def test_ci_install_covers_conftest_imports():
    """Every CI job that runs pytest installs the third-party packages its conftest.py files import."""
    assert_ci_install_covers_conftest(REPO_ROOT)


def test_workflow_paths_exist():
    """A workflow does not name paths that do not exist."""
    assert_workflow_paths_exist(REPO_ROOT / ".github" / "workflows", REPO_ROOT)


def test_default_branch_runs_are_never_cancelled():
    """cancel-in-progress must not be true for a push to the default branch, or main never gets a completed verdict."""
    assert_ci_default_branch_never_cancelled(REPO_ROOT, min_files=1)


def test_comments_name_files_that_exist():
    """A comment naming a tool/, scripts/ or .githooks/ path names a file that exists."""
    assert_comment_repo_paths(REPO_ROOT, min_files=1)


def test_committed_files_have_clean_line_endings():
    """Index blobs with bare CR, mixed CRLF/LF or CRLF endings are findings."""
    assert_committed_line_endings(REPO_ROOT, min_files=1)


def test_coverage_config_does_not_gate_narrow_runs():
    """A whole-suite fail_under inherited by a narrow CI run fails it for coverage it was never meant to reach."""
    assert_coverage_config_parity(REPO_ROOT, min_files=1)


def test_sibling_floors_do_not_exceed_what_ci_installs():
    """A first-party sibling floor in pyproject.toml above the revision the CI workflows install breaks the install."""
    assert_sibling_floor_skew(REPO_ROOT)


def test_commit_subjects_have_no_bom_or_invisible_lead():
    """No commit subject starting with a BOM or an invisible character, and no forbidden trailer."""
    assert_commit_metadata(REPO_ROOT)


def test_tests_that_spawn_git_scrub_the_git_environment():
    """A git child that inherits GIT_DIR from a hook runs against the wrong repository."""
    assert_no_new_unscrubbed_git_spawns(REPO_ROOT)


def test_no_script_reports_skipped_as_a_pass():
    """A script that prints SKIPPED and exits 0 reports a pass."""
    assert_no_new_skip_as_pass(REPO_ROOT)


@pytest.mark.parametrize("baseline", sorted(META.glob("_*baseline*.json")), ids=lambda p: p.name)
def test_baseline_is_a_decision_not_a_landfill(baseline):
    """Every accepted entry of every baseline carries its reason, and none is stale."""
    assert_baseline_is_honest(baseline)
