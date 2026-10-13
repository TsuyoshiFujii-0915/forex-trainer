"""A continuation runtime must name the exact reviewed merge commit."""
from __future__ import annotations

import pytest

from forex_trainer import execution_sensitivity_recovery as recovery


def test_runtime_commit_is_bound_to_the_registration_merge() -> None:
    activation = {'registration_merge_commit': 'a' * 40,
                  'runtime': {'git': {'forex_trainer': 'a' * 40}}}
    recovery.require_activation_commit(activation)
    activation['runtime']['git']['forex_trainer'] = 'b' * 40
    with pytest.raises(ValueError, match='runtime.*merge'):
        recovery.require_activation_commit(activation)
