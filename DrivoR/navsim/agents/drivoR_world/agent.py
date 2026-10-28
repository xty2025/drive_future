"""Bank/future losses added to the original DrivoR training interface."""

from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from .model import DrivoRWorldModel
from .targets import BankTargetBuilder, FutureTargetBuilder


class DrivoRWorldAgent(DrivoRAgent):
    def _build_model(self, config):
        return DrivoRWorldModel(config)

    def get_target_builders(self):
        builders = super().get_target_builders()
        # Restoring a complete inference checkpoint keeps the predictor network,
        # but does not require training-only bank/future-cache files.
        if self._checkpoint_path:
            return builders
        if self._config.bank.enabled:
            builders.append(BankTargetBuilder(self._config.bank))
        if self._config.future.enabled:
            builders.append(FutureTargetBuilder(self._config.future))
        return builders

    def compute_loss(self, features, targets, prediction):
        result = super().compute_loss(features, targets, prediction)
        if self.training and self._config.bank.enabled:
            bank, coverage = self._drivor_model.bank_loss(targets, prediction)
            result["bank_loss"] = bank
            result["bank_scene_coverage"] = coverage
            result["loss"] = result["loss"] + float(self._config.bank.weight) * bank
        if self.training and self._config.future.enabled:
            future = self._drivor_model.realized_future_loss(targets, prediction)
            result["future_loss"] = future
            result["future_target_fraction"] = targets["future_valid"].float().mean()
            result["loss"] = result["loss"] + float(self._config.future.weight) * future
        return result
