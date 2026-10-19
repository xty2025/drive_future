"""Only augment training targets; use the original DrivoR model unchanged."""

from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from .targets import PseudoExpertTargetBuilder


class DrivoRPseudoExpertAgent(DrivoRAgent):
    def get_target_builders(self):
        builders = super().get_target_builders()
        # A full inference checkpoint needs neither the training data nor PE cache.
        if not self._checkpoint_path:
            builders.append(PseudoExpertTargetBuilder(self._config.pseudo_expert))
        return builders
