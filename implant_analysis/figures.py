"""Aggregate-only figures. Everything still requires safe-zone export review."""
import json
from pathlib import Path

from .common import file_sha256


def plot_results(evaluation_dir, training_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(evaluation_dir).resolve()
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    if training_dir and file_sha256(Path(training_dir) / 'last.pt') != metrics['checkpoint_sha256']:
        raise ValueError('평가 모델과 학습 곡선의 체크포인트가 다릅니다.')
    out = directory / 'figures'
    out.mkdir(exist_ok=False)
    matrix = np.asarray(metrics['confusion_matrix'], dtype=float)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix), where=support != 0)
    size = max(6, min(16, len(matrix) * 0.35))
    fig, ax = plt.subplots(figsize=(size, size))
    shown = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    # Class IDs avoid missing Korean fonts; class_map.json provides the labels.
    ticks = list(range(len(matrix)))
    ax.set(xticks=ticks, yticks=ticks, xlabel='Predicted class ID', ylabel='True class ID',
           title=f"{metrics['partition']} | row-normalized confusion matrix")
    if len(matrix) <= 20:
        for i in ticks:
            for j in ticks:
                ax.text(j, i, str(int(matrix[i,j])), ha='center', va='center',
                        color='white' if normalized[i,j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(shown, ax=ax, label='Fraction within true class')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'confusion_matrix.{suffix}', dpi=200)
    plt.close(fig)
    if training_dir:
        history = json.loads((Path(training_dir) / 'history.json').read_text(encoding='utf-8'))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history])
        axes[0].set(xlabel='Epoch', ylabel='Cross-entropy', title='Training loss')
        axes[1].plot([r['epoch'] for r in history], [r['dev_macro_f1'] for r in history])
        axes[1].set(xlabel='Epoch', ylabel='Macro-F1 (all classes)', ylim=(0, 1), title='Development set')
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(out / f'learning_curves.{suffix}', dpi=200)
        plt.close(fig)
    return out
