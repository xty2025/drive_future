#!/usr/bin/env python3
"""Draw actual human trajectory polylines for the existing full/half split."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
COLORS = {'full': '#2476b7', 'half': '#e87935'}
CATEGORIES = ('stationary', 'straight', 'left', 'right')


def limits(points, square=False):
    low = np.minimum(points.min(axis=(0, 1)), 0)
    high = np.maximum(points.max(axis=(0, 1)), 0)
    margin = np.maximum((high - low) * .05, .12 if square else 1.5)
    low, high = low - margin, high + margin
    if square:
        center = (low + high) / 2
        radius = max(high - low) / 2
        low, high = center - radius, center + radius
    return ((float(low[0]), float(high[0])), (float(low[1]), float(high[1])))


def render_trajectory_comparisons(data, half, output, *, seed=2, sample_count=1200, category_count=200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    output = Path(output)
    figure_dir = output / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    trajectory = np.asarray(data['trajectories'])
    half = np.asarray(half, dtype=bool)
    if trajectory.shape[1:] != (8, 3) or half.shape != (len(trajectory),):
        raise ValueError('Expected one half-selection flag per [8,3] human trajectory')
    if not np.isfinite(trajectory).all() or not half.any():
        raise ValueError('Need finite trajectories and a nonempty selected half')
    # The current rear-axle position is known; do not interpolate/smooth the 8 GT points.
    paths = np.concatenate((np.zeros((len(trajectory), 1, 2), dtype=trajectory.dtype), trajectory[..., :2]), axis=1)
    full_ids = np.arange(len(paths))
    half_ids = np.flatnonzero(half)
    rng = np.random.default_rng(seed)
    shared_limits = limits(paths)
    sample_records = []

    def draw(ax, indices, kind, bounds, alpha, width=.6, endpoints=False):
        ax.add_collection(LineCollection(paths[indices], colors=COLORS[kind], linewidths=width, alpha=alpha))
        if endpoints:
            ax.scatter(paths[indices, -1, 0], paths[indices, -1, 1], s=5,
                       color=COLORS[kind], alpha=min(alpha * 1.4, .6), linewidths=0)
        ax.set(xlim=bounds[0], ylim=bounds[1], xlabel='Forward x (m)', ylabel='Left y (m)', aspect='equal')
        ax.set_axisbelow(True)
        ax.grid(alpha=.2)
        ax.scatter([0], [0], marker='o', s=30, facecolors='white', edgecolors='black', linewidths=1, zorder=5)

    def choose(pool, count, plot, kind):
        indices = rng.choice(pool, size=min(count, len(pool)), replace=False)
        for index in indices:
            sample_records.append(dict(plot=plot, source=kind, token=str(data['tokens'][index]),
                log_name=str(data['log_names'][index]), category=str(data['categories'][index]),
                selected_half=bool(half[index])))
        return indices

    def save(fig, name, vector=True):
        figures = []
        for extension in ('png', 'svg') if vector else ('png',):
            relative = f'figures/{name}.{extension}'
            fig.savefig(output / relative, dpi=180)
            figures.append(relative)
        plt.close(fig)
        return figures

    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    full_alpha = .02 * len(half_ids) / len(full_ids)
    for ax, indices, kind, alpha in zip(axes, (full_ids, half_ids), ('full', 'half'), (full_alpha, .02)):
        draw(ax, indices, kind, shared_limits, alpha, width=.35)
        ax.set_title(f'{"All train" if kind == "full" else "Selected half"}: {len(indices):,} actual trajectories')
    fig.suptitle('All human GT trajectories: current origin + 8 future poses (4 s, 2 Hz)', fontsize=14)
    fig.supxlabel('Same metre axes; every trajectory is drawn. Line opacity is scaled for unequal sample counts.', fontsize=10)
    figures += save(fig, 'trajectory_lines_all', vector=False)

    shown = min(sample_count, len(half_ids), len(full_ids))
    sample_full = choose(full_ids, shown, 'equal_count', 'full')
    sample_half = choose(half_ids, shown, 'equal_count', 'half')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)
    draw(axes[0], sample_full, 'full', shared_limits, .11)
    draw(axes[1], sample_half, 'half', shared_limits, .11)
    draw(axes[2], sample_full, 'full', shared_limits, .055)
    draw(axes[2], sample_half, 'half', shared_limits, .055)
    for ax, title in zip(axes, (f'Full pool: {shown:,} shown', f'Half pool: {shown:,} shown', 'Overlay: full + half')):
        ax.set_title(title)
    axes[2].legend(handles=[Line2D([0], [0], color=COLORS[k], label=label)
                           for k, label in [('full', 'Full pool'), ('half', 'Selected half')]], loc='upper right')
    fig.suptitle('Equal display counts reveal individual trajectory shapes', fontsize=14)
    fig.supxlabel(f'Seed {seed} is only for visualization; the selected training tokens are unchanged.', fontsize=10)
    figures += save(fig, 'trajectory_lines_equal_count')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    category_summary = {}
    for column, category in enumerate(CATEGORIES):
        full_pool = np.flatnonzero(data['categories'] == category)
        half_pool = np.flatnonzero((data['categories'] == category) & half)
        shown = min(category_count, len(full_pool), len(half_pool))
        bounds = limits(paths[full_pool], square=True)
        for row, pool, kind, total in [(0, full_pool, 'full', len(full_ids)), (1, half_pool, 'half', len(half_ids))]:
            ids = choose(pool, shown, 'category_' + category, kind)
            ax = axes[row, column]
            draw(ax, ids, kind, bounds, .20, width=.7, endpoints=True)
            ax.set_title(f'{category.title()} — {"full" if row == 0 else "half"}\n'
                         f'{len(pool):,} / {total:,} ({len(pool) / total * 100:.3f}%), {shown} shown', fontsize=11)
        category_summary[category] = dict(full_count=len(full_pool), half_count=len(half_pool),
                                           shown_each=shown, limits=bounds)
    fig.suptitle('Human trajectory groups: full (top) vs selected half (bottom)', fontsize=15)
    fig.supxlabel('Same scale within each column; equal metres on x/y; stationary column is zoomed in. Dots mark the 4 s endpoint.', fontsize=10)
    figures += save(fig, 'trajectory_lines_by_category')

    with (output / 'trajectory_visualization_samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['plot', 'source', 'token', 'log_name', 'category', 'selected_half'])
        writer.writeheader()
        writer.writerows(sample_records)
    result = dict(created_at=datetime.now().astimezone().isoformat(), full_count=len(full_ids), half_count=len(half_ids),
        poses_per_trajectory=8, horizon_seconds=4, frequency_hz=2, coordinate_frame='current_ego_rear_axle',
        smoothing=False, all_plot_limits=shared_limits, full_alpha=full_alpha, half_alpha=.02,
        equal_count_shown=min(sample_count, len(half_ids)), display_seed=seed, category_groups=category_summary,
        figures=figures, selected_training_tokens_changed=False, used_gpu=False)
    (output / 'trajectory_visualization.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    text = f'''# 半量数据：真实 human 轨迹可视化

全部 {len(full_ids):,} 条训练 human GT 与已选 {len(half_ids):,} 条对照。每条曲线连接当前原点和原始 8 个未来点，覆盖 4 秒、2 Hz；单位为米，自车后轴坐标系，横轴前方、纵轴左方。没有生成新轨迹或平滑插值。

## 全部轨迹

![全部轨迹对照](figures/trajectory_lines_all.png)

左侧绘制全部 {len(full_ids):,} 条，右侧绘制保留的 {len(half_ids):,} 条。两图使用同一坐标范围，覆盖全部原始轨迹点；线条透明度按样本数调整以便观察。叠线深浅不是严格概率密度，精确分布仍看原来的归一化热图和统计。

## 等量展示与叠加

![等量轨迹对照](figures/trajectory_lines_equal_count.png)

分别从全量和半量池抽 {min(sample_count, len(half_ids)):,} 条用于展示，第三列叠加两组曲线。蓝色是全量池，橙色是半量池。出图随机种子 {seed} 只影响展示，不修改实际训练 token。

## 按运动类别对照

![分组轨迹对照](figures/trajectory_lines_by_category.png)

上排全量、下排半量，每列两图坐标范围相同，最多各展示 {category_count} 条。圆点表示 +4 秒终点，原点空心点表示当前自车。静止列单独放大。分类沿用原分布报告：4 秒路程小于 1 米为 stationary；其余按最终航向变化大于 +15°、小于 −15°、其他划分为 left、right、straight。

全部分布统计见 [REPORT.md](REPORT.md)。等量/分组展示所用场景 token 见 [trajectory_visualization_samples.csv](trajectory_visualization_samples.csv)，参数见 [trajectory_visualization.json](trajectory_visualization.json)。

复现：`python DrivoR/tools/plot_navtrain_half_trajectories.py`。仅读取已有 human GT 缓存和半量 token 文件，不读取图像、不训练模型、不访问 DLC。等量和分组图同时提供 SVG 文件。
'''
    (output / 'TRAJECTORY_VISUALIZATIONS.md').write_text(text)
    return result


def report_section():
    return '''<!-- DIRECT_TRAJECTORY_VISUALIZATIONS -->
### 原始轨迹曲线

![全部真实轨迹对照](figures/trajectory_lines_all.png)

[等量轨迹叠加](figures/trajectory_lines_equal_count.png) · [直行/转弯/静止分组](figures/trajectory_lines_by_category.png) · [完整轨迹可视化说明](TRAJECTORY_VISUALIZATIONS.md)

每条线连接当前原点和原始 8 个未来 GT 点。等量与分类图的抽样仅用于显示，实际半量训练 tokens 不变。
<!-- END_DIRECT_TRAJECTORY_VISUALIZATIONS -->

'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'DrivoR/exp/navtrain_half')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--sample-count', type=int, default=1200)
    parser.add_argument('--category-count', type=int, default=200)
    args = parser.parse_args()
    if args.sample_count < 1 or args.category_count < 1:
        parser.error('Display sample counts must be positive')
    with np.load(args.data_dir / 'human_trajectory_features.npz', allow_pickle=False) as stored:
        data = {name: stored[name] for name in ('tokens', 'trajectories', 'categories', 'log_names')}
    selected = (args.data_dir / 'selected_train_tokens.txt').read_text().splitlines()
    selected_set = set(selected)
    if len(selected_set) != len(selected) or not selected_set <= set(data['tokens'].tolist()):
        raise ValueError('Selected training tokens must be a unique subset of the cached human GT')
    half = np.array([token in selected_set for token in data['tokens']], dtype=bool)
    result = render_trajectory_comparisons(data, half, args.data_dir, seed=args.seed,
                                          sample_count=args.sample_count, category_count=args.category_count)
    report = args.data_dir / 'REPORT.md'
    if report.exists():
        import re
        content = re.sub(r'<!-- DIRECT_TRAJECTORY_VISUALIZATIONS -->.*?<!-- END_DIRECT_TRAJECTORY_VISUALIZATIONS -->\n*',
                         '', report.read_text(), flags=re.S)
        marker = '## 检查结论与范围'
        content = content.replace(marker, report_section() + marker, 1) if marker in content else content + '\n' + report_section()
        report.write_text(content)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
