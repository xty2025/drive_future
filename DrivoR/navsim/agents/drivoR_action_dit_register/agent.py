"""Training agent with EMA updates after actual optimizer steps and staged loading."""

import logging

from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.agents.drivoR.multi_gt import multi_gt_enabled
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_pseudo_expert.targets import PseudoExpertTargetBuilder

from .model import DrivoRActionRegisterModel
from .targets import FutureRegisterTargetBuilder, HumanFutureRegisterTargetBuilder


log = logging.getLogger(__name__)


class DrivoRActionRegisterAgent(DrivoRActionDiTAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        path = self._config.future_register.get('init_checkpoint')
        if path and not self._checkpoint_path:
            payload = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            state = {key.removeprefix('agent.'):value for key,value in payload['state_dict'].items()}
            self.load_state_dict(state,strict=True)
            log.info('Loaded Stage1 model and EMA teacher from %s; Stage2 uses a new optimizer',path)

    def _build_model(self, config):
        return DrivoRActionRegisterModel(config)

    def initialize(self):
        if self._checkpoint_path:
            payload = torch.load(self._checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
            self.load_state_dict({key.removeprefix('agent.'):value for key,value in payload['state_dict'].items()},strict=True)

    @property
    def lightning_module_class(self):
        from .lightning import RegisterLightningModule
        return RegisterLightningModule

    def get_target_builders(self):
        builders = [DrivoRTargetBuilder(self._config)]
        if not self._checkpoint_path:
            if multi_gt_enabled(self._config):
                builders.append(FutureRegisterTargetBuilder(self._config.multi_gt,self._config.future_register))
            else:
                builders.append(HumanFutureRegisterTargetBuilder(self._config.future_register))
            if (self._drivor_model.future_stage == 'joint'
                    and self._config.get('pseudo_expert', {}).get('enabled', False)):
                builders.append(PseudoExpertTargetBuilder(self._config.pseudo_expert))
        return builders

    def compute_loss(self, features, targets, pred):
        future = pred['_register_loss']
        weight = float(self._config.future_register.loss_weight)
        if self._drivor_model.future_stage == 'future_only':
            result = {'loss':weight*future,'trajectory_loss':future.detach()*0.,'final_score_loss':future.detach()*0.}
            result.update(pred['_flow_diagnostics'])
        else:
            result = super().compute_loss(features,targets,pred)
            result['loss'] = result['loss']+weight*future
            if self._config.get('pseudo_expert', {}).get('enabled', False):
                expert_loss = pred['_pseudo_expert_velocity_loss']
                added = float(self._config.pseudo_expert.weight)*expert_loss
                result['pseudo_expert_loss'] = expert_loss
                result['pseudo_expert_count'] = targets['pseudo_expert_mask'].sum(-1).float().mean()
                result['pseudo_expert_fallback_fraction'] = targets['pseudo_expert_fallback'].float().mean()
                result['trajectory_loss'] = result['trajectory_loss']+added
                result['loss'] = result['loss']+self.loss.trajectory_weight*added
        result['future_register_loss'] = future
        return result

    def get_optimizers(self):
        configured = super().get_optimizers()
        optimizers = configured[0] if isinstance(configured,tuple) else configured
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['params'] = [parameter for parameter in group['params'] if parameter.requires_grad]
            optimizer.register_step_post_hook(self._after_optimizer_step)
        return configured

    @torch.no_grad()
    def _after_optimizer_step(self, optimizer, args, kwargs):
        model = self._drivor_model
        model.ema_teacher.update(model.image_backbone.native_vit)

    def get_training_callbacks(self):
        callbacks = super().get_training_callbacks()
        if self._drivor_model.future_stage == 'future_only':
            callbacks = [callback for callback in callbacks
                         if not isinstance(callback,ModelCheckpoint) or callback.monitor is None]
        return callbacks
