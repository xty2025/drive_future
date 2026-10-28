"""Real video frame selection and slot-preserving, reusable RGB caches."""

import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from omegaconf import OmegaConf
import pytest

from navsim.common import diverse_future_frames as frames
from navsim.agents.drivoR_action_dit_register.diverse_targets import DiverseFutureRegisterTargetBuilder


def sources(tmp_path):
    token = '0123456789abcdef'
    gt, video, cache = (tmp_path / name for name in ('gt', 'video', 'cache'))
    folder = gt / token; folder.mkdir(parents=True)
    poses = np.zeros((3, 8, 3), np.float32); poses[2] = 2
    np.save(folder/'gt_trajectories.npy', poses)
    np.save(folder/'gt_valid_mask.npy', [True, True, True])
    np.save(folder/'gt_like_mask.npy', [True, False, False])
    (folder/'complete.json').write_text('{}')
    info = dict(gt_padded=[False, True, False], gt_labels=['first', 'padding', 'third'],
                selected_indices=[11, 22, 33], target_times=frames.TIMES.tolist())
    (folder/'trajectory.json').write_text(json.dumps(info))
    (video/token).mkdir(parents=True)
    (video/token/'completion.json').write_text(json.dumps(dict(status='COMPLETE')))
    for slot in (0, 2):
        target = video/token/f'trajectory_{slot}'; target.mkdir()
        (target/'metadata.json').write_text(json.dumps(dict(token=token, valid=True, source_valid=True,
            source_padded=False, trajectory_index=slot, trajectory_label=info['gt_labels'][slot],
            source_selected_candidate_index=info['selected_indices'][slot])))
        np.save(target/'pose_times_s.npy', np.arange(41)*.1)
        # Every source frame has a known intensity, making t=0 leakage observable.
        rgb = np.broadcast_to((np.arange(41)*5+slot).astype(np.uint8)[:,None,None,None],(41,16,16,3)).copy()
        subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','16x16','-r','10',
            '-i','pipe:0','-c:v','libx264rgb','-qp','0','-threads','1',str(target/'future.mp4')],
            input=rgb.tobytes(),capture_output=True,check=True)
    reader = DiverseFutureRegisterTargetBuilder(
        OmegaConf.create(dict(path=str(gt),top_k=8,missing_policy='ignore')),
        OmegaConf.create(dict(frames_root=str(cache),rollouts_root=str(video),image_size=[16,16],
                              sensor_blobs_path='unused',build_missing_cache=True)))
    return token, gt, video, cache, reader


def test_real_video_indices_sparse_slots_and_cache_reuse(tmp_path,monkeypatch):
    token,gt,video,cache,reader=sources(tmp_path)
    result=reader.load(token)
    assert result['pseudo_expert_mask'].tolist()==[True,False,True]+[False]*5
    assert result['future_register_valid'].any(-1).tolist()==[True,False,True]+[False]*5
    for slot in (0,2):
        for time,index in enumerate(frames.FRAME_INDICES):
            assert abs(result['future_register_images'][slot,time].float().mean().item()-(index*5+slot))<=1
    assert not result['future_register_images'][1].any()
    archive=cache/token[:2]/(token+'.zip')
    with zipfile.ZipFile(archive) as z:
        metadata=json.loads(z.read('metadata.json'))
    assert [entry['slot'] for entry in metadata['slots']]==[0,2]
    assert metadata['slots'][1]['candidate_index']==33
    monkeypatch.setattr(frames,'decode_future_frames',lambda *args:(_ for _ in ()).throw(AssertionError('Cached read decoded video')))
    again=reader.load(token)
    assert np.array_equal(result['future_register_images'],again['future_register_images'])
    info=json.loads((gt/token/'trajectory.json').read_text());info['selected_indices'][2]=99
    (gt/token/'trajectory.json').write_text(json.dumps(info))
    with pytest.raises(ValueError,match='pairing mismatch'):
        reader.load(token)


def test_missing_video_is_an_error_not_silent_human_only_alignment(tmp_path):
    token,gt,video,cache,reader=sources(tmp_path)
    (video/token/'trajectory_2'/'future.mp4').unlink()
    with pytest.raises(RuntimeError,match='Cannot decode'):
        reader.load(token)
    assert not (cache/token[:2]/(token+'.zip')).exists()
    assert not list((cache/token[:2]).glob('*.tmp'))


def test_no_diverse_gt_keeps_empty_future_slots_without_video_lookup(tmp_path):
    reader=DiverseFutureRegisterTargetBuilder(OmegaConf.create(dict(path=str(tmp_path),top_k=8,missing_policy='ignore')),
        OmegaConf.create(dict(frames_root='/absent/cache',rollouts_root='/absent/videos',image_size=[16,16],sensor_blobs_path='unused')))
    result=reader.load('0000000000000000')
    assert result['diverse_gt_missing'] and not result['future_register_valid'].any()


def test_offline_empty_gt_scene_does_not_require_a_generated_video(tmp_path):
    token,gt,video,cache,reader=sources(tmp_path)
    np.save(gt/token/'gt_valid_mask.npy',[False,False,False])
    (video/token/'completion.json').unlink()
    path=frames.extract_diverse_scene_frames(token,gt,video,cache,(16,16))
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read('metadata.json'))['slots']==[]


@pytest.mark.parametrize('link_kind', ['directory', 'files'])
@pytest.mark.parametrize('cache_origin', ['merged', 'supplement'])
def test_linked_gt_keeps_slot_alignment_and_reuses_cache(tmp_path, monkeypatch, link_kind, cache_origin):
    token, gt, video, cache, original = sources(tmp_path)
    merged = tmp_path / 'merged'
    merged.mkdir()
    if link_kind == 'directory':
        (merged / token).symlink_to(gt / token, target_is_directory=True)
    else:
        (merged / token).mkdir()
        for source in (gt / token).iterdir():
            (merged / token / source.name).symlink_to(source)
    linked = DiverseFutureRegisterTargetBuilder(
        OmegaConf.create(dict(path=str(merged), top_k=8, missing_policy='error')),
        OmegaConf.create(dict(frames_root=str(cache), rollouts_root=str(video),
                              image_size=[16, 16], sensor_blobs_path='unused')))
    producer = linked if cache_origin == 'merged' else original
    expected = producer.load(token)
    monkeypatch.setattr(frames, 'decode_future_frames',
                        lambda *args: (_ for _ in ()).throw(AssertionError('Re-decoded a valid cached video')))
    actual = linked.load(token)
    assert actual['pseudo_expert_mask'].tolist() == [True, False, True] + [False] * 5
    assert actual['future_register_valid'].any(-1).tolist() == [True, False, True] + [False] * 5
    assert np.array_equal(actual['pseudo_experts'], expected['pseudo_experts'])
    assert np.array_equal(actual['future_register_images'], expected['future_register_images'])
    # Accept only the merged path or the GT file's actual source root.
    archive_path = cache / token[:2] / (token + '.zip')
    with zipfile.ZipFile(archive_path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(contents['metadata.json'])
    metadata['source_gt_root'] = str(tmp_path / 'unrelated')
    contents['metadata.json'] = json.dumps(metadata).encode()
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match='source/geometry mismatch'):
        linked.load(token)
