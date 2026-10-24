"""Choose last.ckpt and, for native Wan 5B ZeRO-2, export an FP32 eval file."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys


def prepare_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    destination = checkpoint.with_name(checkpoint.stem + '.fp32.ckpt')
    temporary = destination.with_suffix('.tmp')
    # DeepSpeed writes a directory per epoch. Export only the final checkpoint
    # for the existing ordinary FP32 evaluation loader; keep all training shards.
    with redirect_stdout(sys.stderr):
        import torch
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, get_model_state_file
        state = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint))
        tag = (checkpoint / 'latest').read_text().strip()
        # These are locally generated training checkpoints, including optimizer
        # objects. PL 2.4's converter omits weights_only=False and fails with
        # PyTorch >=2.6; load the small client metadata explicitly here instead.
        model_file = get_model_state_file(str(checkpoint / tag), zero_stage=2)
        client = torch.load(model_file, map_location='cpu', weights_only=False, mmap=True)
        payload = {key: client[key] for key in ('epoch', 'global_step', 'pytorch-lightning_version',
                                               'future_register_stage') if key in client}
        payload['state_dict'] = {key.removeprefix('_forward_module.'): value for key, value in state.items()}
        torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir', type=Path)
    group.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if args.run_dir:
        paths = list(args.run_dir.glob('lightning_logs/version_*/checkpoints/last.ckpt'))
        if not paths:
            raise FileNotFoundError(f'No last.ckpt under {args.run_dir}')
        checkpoint = max(paths, key=lambda path: path.stat().st_mtime_ns)
    print(prepare_checkpoint(checkpoint))


if __name__ == '__main__':
    main()
