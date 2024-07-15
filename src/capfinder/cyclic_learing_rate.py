from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from comet_ml import Experiment
from numpy.typing import NDArray

from capfinder.ml_libs import Callback, keras


class CometLRLogger(Callback):
    """
    A callback to log the learning rate to Comet.ml during training.

    This callback logs the learning rate at the beginning of each epoch
    and at the end of each batch to a Comet.ml experiment.

    Attributes:
        experiment (Experiment): The Comet.ml experiment to log to.
    """

    def __init__(self, experiment: Experiment) -> None:
        """
        Initialize the CometLRLogger.

        Args:
            experiment (Experiment): The Comet.ml experiment to log to.
        """
        super().__init__()
        self.experiment: Experiment = experiment

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the learning rate at the beginning of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (Optional[Dict[str, Any]]): The logs dictionary.
        """
        lr: Union[float, np.ndarray] = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        self.experiment.log_metric("learning_rate", lr, step=epoch)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the learning rate at the end of each batch.

        Args:
            batch (int): The current batch number.
            logs (Optional[Dict[str, Any]]): The logs dictionary.
        """
        lr: Union[float, np.ndarray] = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        self.experiment.log_metric(
            "learning_rate", lr, step=self.model.optimizer.iterations.numpy()
        )


class CustomProgressCallback(keras.callbacks.Callback):
    """
    A custom callback to print the learning rate at the end of each epoch.

    This callback prints the current learning rate after Keras' built-in
    progress bar for each epoch.
    """

    def __init__(self) -> None:
        """Initialize the CustomProgressCallback."""
        super().__init__()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Print the learning rate at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (Optional[Dict[str, Any]]): The logs dictionary.
        """
        lr: Union[float, np.ndarray] = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        print(f"\nLearning rate: {lr:.6f}")


class CyclicLR(Callback):
    """
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.

    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(
        self,
        base_lr: float = 0.001,
        max_lr: float = 0.006,
        step_size: float = 2000.0,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[float], float]] = None,
        scale_mode: str = "cycle",
    ) -> None:
        super().__init__()

        self.base_lr: float = base_lr
        self.max_lr: float = max_lr
        self.step_size: float = step_size
        self.mode: str = mode
        self.gamma: float = gamma

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma**x
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.clr_iterations: float = 0.0
        self.trn_iterations: float = 0.0
        self.history: Dict[str, list] = {}

        self._reset()

    def _reset(
        self,
        new_base_lr: Optional[float] = None,
        new_max_lr: Optional[float] = None,
        new_step_size: Optional[float] = None,
    ) -> None:
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def clr(self) -> Union[float, NDArray[np.float64]]:
        cycle: float = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x: float = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        clr_value: float = (
            self.base_lr
            + (self.max_lr - self.base_lr)
            * np.maximum(0, (1 - x))
            * self.scale_fn(cycle)
            if self.scale_mode == "cycle"
            else self.scale_fn(self.clr_iterations)
        )
        return (
            float(clr_value)
            if isinstance(clr_value, float)
            else np.array(clr_value, dtype=np.float64)
        )

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the learning rate to the base learning rate."""
        logs = logs or {}

        if self.clr_iterations == 0:
            self.model.optimizer.learning_rate.assign(self.base_lr)
        else:
            self.model.optimizer.learning_rate.assign(self.clr())

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault("lr", []).append(
            self.model.optimizer.learning_rate.numpy()
        )
        self.history.setdefault("iterations", []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.model.optimizer.learning_rate.assign(self.clr())
