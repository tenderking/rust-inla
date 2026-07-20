"""Test Python signal interrupt handling (SIGINT / KeyboardInterrupt) during INLA inference."""

import signal
import threading
import time
import pytest
import numpy as np
import inla
from inla._native import run_inla_inference


def test_interrupt_handling():
    """Verify that PyO3 check_signals hook interrupts long-running inference with KeyboardInterrupt."""
    n = 200
    y = np.random.randn(n)

    # Slow custom prior callback that simulates slow work or allows signal arrival
    def slow_prior(theta):
        time.sleep(0.1)
        return inla.iid_precision_matrix(n, 1.0)

    def trigger_sigint():
        time.sleep(0.01)
        signal.raise_signal(signal.SIGINT)

    # Start a thread to trigger SIGINT shortly after inference starts
    timer = threading.Thread(target=trigger_sigint)
    timer.start()

    with pytest.raises(KeyboardInterrupt):
        run_inla_inference(
            initial_theta=[0.0],
            build_prior=slow_prior,
            log_prior_density=lambda theta: -0.5 * theta[0] ** 2,
            obs=[{"family": "gaussian", "y": float(yi), "precision": 1.0} for yi in y],
            strategy="grid",
            step_or_f0=0.5,
            n_points=50,
        )

    timer.join()
