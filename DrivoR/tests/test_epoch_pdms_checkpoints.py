"""Epoch retention, true validation PDMS, FP32 inference and sampler padding."""

import csv

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from navsim.planning.script.run_training import build_dataloader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.validation_pdms import summarize_scene_pdms, write_epoch_pdms


def test_validation_keeps_incomplete_batch_and_training_default_is_unchanged():
    dataset = TensorDataset(torch.arange(5))
    options = dict(batch_size=2,num_workers=0)
    train = build_dataloader(dataset,options,shuffle=True,seed=2)
    val = build_dataloader(dataset,options,shuffle=False,seed=3)
    assert train.drop_last and len(train)==2
    assert not val.drop_last and [batch[0].tolist() for batch in val]==[[0,1],[2,3],[4]]


def test_padding_scenes_are_counted_once_and_rank_order_is_deterministic():
    result = summarize_scene_pdms([{'a':.2,'c':.9},{'b':.7,'a':.8}])
    assert result['val_scenes']==3
    assert result['val_pdms']==pytest.approx(.6)
    assert summarize_scene_pdms([{},{}]) is None


def test_epoch_csv_replaces_a_repeated_epoch_on_resume(tmp_path):
    write_epoch_pdms(tmp_path,0,10,{'val_scenes':3,'val_pdms':.4},'32-true')
    path = write_epoch_pdms(tmp_path,1,20,{'val_scenes':3,'val_pdms':.6},'32-true')
    write_epoch_pdms(tmp_path,1,20,{'val_scenes':3,'val_pdms':.7},'32-true')
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['epoch'] for row in rows]==['0','1']
    assert float(rows[-1]['val_pdms'])==.7 and rows[-1]['precision']=='32-true'


class TinyScenes(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self,index):
        return ({'input':torch.tensor([float(index+1)])},
                {'token':f'scene_{index}','trajectory':torch.zeros(8,3),
                 'oracle_quality':torch.tensor([.2,.6,1.][index])})


class DrivoRTinyAgent(nn.Module):
    """Synthetic Oracle values verify runtime plumbing, not NAVSIM quality."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1,1)
        self.forward_dtypes = []
        self.progress_bar = True

    def name(self):
        return 'DrivoRTinyAgent'

    def forward_train(self,features,targets):
        return {'loss':self.layer(features['input']).square().mean()}

    def compute_loss(self,features,targets,prediction):
        return prediction

    def forward(self,features):
        value = self.layer(features['input'])
        self.forward_dtypes.append(value.dtype)
        proposals = value[:,None,None,:].expand(-1,5,8,3)
        return {'trajectory':proposals[:,4], 'proposals':proposals,
                'pdm_score':torch.arange(5.,device=value.device)[None].expand(len(value),-1)}

    def compute_score(self,targets,proposals,test=True):
        score = targets['oracle_quality'][:,None].expand(*proposals.shape[:2])
        raw7 = score[:,0:1].expand(-1,7)
        return score[:,0].mean(),score.amax(-1).mean(),score,score.new_zeros(()),raw7

    def get_optimizers(self):
        return [torch.optim.AdamW(self.parameters(),lr=.001)]


def test_two_epochs_keep_all_checkpoints_and_record_fp32_oracle_pdms(tmp_path):
    agent = DrivoRTinyAgent()
    module = AgentLightningModule(agent,console_log_interval=0,log_local_steps=False)
    callbacks = [callback for callback in DrivoRAgent.get_training_callbacks(agent)
                 if isinstance(callback,ModelCheckpoint)]
    every_epoch = next(callback for callback in callbacks if callback.monitor is None)
    assert every_epoch.save_top_k==-1 and every_epoch.save_last
    trainer = pl.Trainer(accelerator='cpu',devices=1,precision='bf16-mixed',max_epochs=2,
        check_val_every_n_epoch=1,val_check_interval=1.,num_sanity_val_steps=0,
        default_root_dir=tmp_path,logger=CSVLogger(tmp_path/'logs'),callbacks=callbacks,
        enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
    params = dict(batch_size=2,num_workers=0)
    trainer.fit(module,
        build_dataloader(TinyScenes(),params,shuffle=True,seed=2),
        build_dataloader(TinyScenes(),params,shuffle=False,seed=3))
    assert set(agent.forward_dtypes)=={torch.float32}
    checkpoint_dir = every_epoch.dirpath
    from pathlib import Path
    files = list(Path(checkpoint_dir).glob('epoch=*-step=*.ckpt'))
    assert len(files)==2
    for epoch,path in enumerate(sorted(files)):
        checkpoint = torch.load(path,weights_only=False,map_location='cpu')
        assert checkpoint['epoch']==epoch and checkpoint['global_step']==epoch+1
        assert checkpoint['optimizer_states']
    assert (Path(checkpoint_dir)/'last.ckpt').is_file()
    assert len(list(Path(checkpoint_dir).glob('best-*.ckpt')))==1
    with (tmp_path/'val_pdms.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['epoch'] for row in rows]==['0','1']
    assert all(int(row['val_scenes'])==3 and row['precision']=='32-true' for row in rows)
    assert all(float(row['val_pdms'])==pytest.approx(.6) for row in rows)
    assert float(trainer.callback_metrics['val/pdms'])==pytest.approx(.6)
    assert torch.equal(trainer.callback_metrics['val/pdms'],trainer.callback_metrics['val/score_epoch'])
