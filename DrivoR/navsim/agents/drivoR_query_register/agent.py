"""Reuse DrivoR runtime and EMA lifecycle with the non-diffusion generator."""

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.agents.drivoR.multi_gt import multi_gt_enabled
from navsim.agents.drivoR_action_dit_register.agent import DrivoRActionRegisterAgent
from navsim.agents.drivoR_action_dit_register.targets import HumanFutureRegisterTargetBuilder

from .model import DrivoRQueryRegisterModel
from .targets import QueryFutureRegisterTargetBuilder


class DrivoRQueryRegisterAgent(DrivoRActionRegisterAgent):
    def _build_model(self, config):
        return DrivoRQueryRegisterModel(config)

    def get_target_builders(self):
        builders = [DrivoRTargetBuilder(self._config)]
        if not self._checkpoint_path:
            if multi_gt_enabled(self._config):
                builders.append(QueryFutureRegisterTargetBuilder(self._config.multi_gt, self._config.future_register))
            else:
                builders.append(HumanFutureRegisterTargetBuilder(self._config.future_register))
        return builders

    def compute_loss(self, features, targets, pred):
        return self.loss(targets, pred, self._config, self.compute_score)
