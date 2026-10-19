"""Stage1 validates register denoising without invoking the Scorer or NAVSIM oracle."""

from navsim.planning.training.agent_lightning_module import AgentLightningModule, _batch_size


class RegisterLightningModule(AgentLightningModule):
    def on_save_checkpoint(self, checkpoint):
        checkpoint['future_register_stage'] = self.agent._drivor_model.future_stage

    def validation_step(self, batch, batch_idx):
        if self.agent._drivor_model.future_stage != 'future_only':
            return super().validation_step(batch,batch_idx)
        features,targets = batch
        prediction = self.agent.forward_train(features,targets)
        losses = self.agent.compute_loss(features,targets,prediction)
        batch_size = _batch_size(features,targets)
        for name,value in losses.items():
            self.log('val/'+name,value,on_step=False,on_epoch=True,sync_dist=True,batch_size=batch_size)
        return losses['loss']
