"""Optional Bank or CLOVER coverage losses on the existing Action-DiT model.

The two launch configurations enable one auxiliary objective each. Both keep
the original multi-GT v loss and the single shared X-prediction training pass.
"""

from navsim.agents.drivoR_world.model import sample_bank, bank_outcome_loss
from navsim.agents.drivoR_world.targets import BankTargetBuilder
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss
from navsim.agents.drivoR_pseudo_expert.targets import PseudoExpertTargetBuilder

from .agent import DrivoRActionDiTAgent


class DrivoRActionDiTAuxiliaryAgent(DrivoRActionDiTAgent):
    def get_target_builders(self):
        builders = super().get_target_builders()
        if self._checkpoint_path:
            return builders
        if self._config.get("bank", {}).get("enabled", False):
            builders.append(BankTargetBuilder(self._config.bank))
        if self._config.get("pseudo_expert", {}).get("enabled", False):
            builders.append(PseudoExpertTargetBuilder(self._config.pseudo_expert))
        return builders

    def compute_loss(self, features, targets, pred):
        result = super().compute_loss(features, targets, pred)
        if not self.training:
            return result
        if self._config.get("bank", {}).get("enabled", False):
            options = self._config.bank
            candidates, labels, valid = sample_bank(
                targets["bank_candidates"], targets["bank_subscores"], targets["bank_valid"],
                count=int(options.count), safety_fraction=float(options.safety_fraction))
            # Reuse current DINOv3 patches and ego embedding. Bank candidates
            # use the same scorer weights, in a separate 16-token attention set.
            bank_prediction = self._drivor_model.score_proposals(
                candidates, pred["_scene_context"], pred["_ego_embedding"])
            loss = bank_outcome_loss(bank_prediction["pred_logit"], labels, valid)
            result["bank_loss"] = loss
            result["bank_scene_coverage"] = valid.any(-1).float().mean()
            result["loss"] = result["loss"] + float(options.weight) * loss
        if self._config.get("pseudo_expert", {}).get("enabled", False):
            if self._config.pseudo_expert.get("loss_type", "l1") == "velocity":
                loss = pred["_pseudo_expert_velocity_loss"]
            else:
                loss = multi_expert_coverage_loss(
                    pred["_pseudo_expert_proposals"], targets["pseudo_experts"], targets["pseudo_expert_mask"])
            added = float(self._config.pseudo_expert.weight) * loss
            result["pseudo_expert_loss"] = loss
            result["pseudo_expert_count"] = targets["pseudo_expert_mask"].sum(-1).float().mean()
            result["pseudo_expert_fallback_fraction"] = targets["pseudo_expert_fallback"].float().mean()
            result["trajectory_loss"] = result["trajectory_loss"] + added
            result["loss"] = result["loss"] + self.loss.trajectory_weight * added
        return result
