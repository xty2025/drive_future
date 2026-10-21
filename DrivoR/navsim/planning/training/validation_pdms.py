"""Scene-level validation PDMS and a compact, resume-friendly epoch history."""

import csv
import math
from pathlib import Path


def summarize_scene_pdms(rank_records):
    # DistributedSampler pads its final shard by repeating scene tokens.
    # Use the first occurrence (rank order) so each scene contributes once.
    scenes = {}
    for records in rank_records:
        for token, score in records.items():
            scenes.setdefault(token, float(score))
    if not scenes:
        return None
    return {'val_scenes': len(scenes), 'val_pdms': math.fsum(scenes.values())/len(scenes)}


def write_epoch_pdms(output_dir, epoch, global_step, summary, precision):
    path = Path(output_dir)/'val_pdms.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    rows = {}
    if path.exists():
        with path.open(newline='') as handle:
            rows = {int(row['epoch']):row for row in csv.DictReader(handle)}
    rows[int(epoch)] = {'epoch':int(epoch), 'global_step':int(global_step),
                       **summary, 'precision':precision}
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['epoch','global_step','val_scenes','val_pdms','precision'])
        writer.writeheader()
        writer.writerows(rows[index] for index in sorted(rows))
    temporary.replace(path)
    return path
