from typing import List, Optional

from avalanche.evaluation import GenericPluginMetric, Metric, PluginMetric
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch import Tensor


class FLOPS(Metric[int]):
    """
    Measuring the number of FLOPS in a forward pass of the model.
    """

    def __init__(self):
        """
        Creates an instance of the FLOPS metric.
        """
        self._flops = 0
        self._profiler = None

    def update(self, strategy, dummy_input: Tensor) -> None:
        """
        Update the metric value.
        """

        self._profiler = FlopsProfiler(strategy.model)
        if glob_profiler_step == strategy.clock.train_exp_iterations:
            self._profiler.start_profile()
            strategy.model(dummy_input)

            self._flops = self._profiler.get_total_flops()
            self._profiler.end_profile()

    def result(self) -> Optional[int]:
        """
        Returns the current value of the metric.
        """
        return self._flops

    def reset(self) -> None:
        pass


class FLOPSPluginMetric(GenericPluginMetric):
    def __init__(self, reset_at, emit_at, mode):
        self._flop = FLOPS()

        super(FLOPSPluginMetric, self).__init__(
            self._flop, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def update(self, strategy):
        self._flop.update(strategy, strategy.mb_x[0].unsqueeze(0))


class MinibatchFLOPS(FLOPSPluginMetric):
    """
    The FLOPS measured at the end of a given mini-batch controled by the
    `profiler_step` parameter.
    """

    def __init__(self):
        super(MinibatchFLOPS, self).__init__(
            reset_at="iteration",
            emit_at="iteration",
            mode="train",
        )

    def before_training_iteration(self, strategy):
        super().before_training_iteration(strategy)
        if strategy.clock.train_exp_iterations == glob_profiler_step:
            print("Reached")
        self._flop.update(strategy, strategy.mb_x[0].unsqueeze(0))
        return None

    def after_training_iteration(self, strategy):
        super().after_training_iteration(strategy)
        self._flop.result()
        return self._package_result(strategy)

    def __str__(self):
        return f"FLOPS_MB{glob_profiler_step}"


def flops_metrics(*, minibatch=False, profiler_step=10) -> List[PluginMetric]:
    metrics: List[FLOPSPluginMetric] = []
    global glob_profiler_step
    glob_profiler_step = profiler_step
    if minibatch:
        metrics.append(MinibatchFLOPS())
    return metrics


__all__ = ["MinibatchFLOPS", "flops_metrics"]
