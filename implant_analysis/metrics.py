"""Fixed-label metrics; absent classes are explicit rather than silently dropped."""
import math


def classification_metrics(targets, probabilities, class_count):
    if not targets or len(targets) != len(probabilities):
        raise ValueError('라벨과 예측의 길이를 확인하세요.')
    matrix = [[0] * class_count for _ in range(class_count)]
    top3 = 0
    for target, probs in zip(targets, probabilities):
        if len(probs) != class_count or not 0 <= target < class_count or \
                any(not math.isfinite(p) or p < 0 or p > 1 for p in probs) or abs(sum(probs) - 1) > 1e-4:
            raise ValueError('유효하지 않은 라벨 또는 확률입니다.')
        order = sorted(range(class_count), key=lambda i: (-probs[i], i))
        matrix[target][order[0]] += 1
        top3 += target in order[:min(3, class_count)]
    per_class = []
    for k in range(class_count):
        tp = matrix[k][k]
        support = sum(matrix[k])
        predicted = sum(row[k] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(dict(class_id=k, support=support, precision=precision, recall=recall, f1=f1))
    supported = [r for r in per_class if r['support']]
    return {
        'n': len(targets), 'accuracy': sum(matrix[k][k] for k in range(class_count)) / len(targets),
        'top_k': min(3, class_count), 'top_k_accuracy': top3 / len(targets),
        'macro_f1_all_classes': sum(r['f1'] for r in per_class) / class_count,
        'macro_f1_supported_classes': sum(r['f1'] for r in supported) / len(supported),
        'balanced_accuracy_supported_classes': sum(r['recall'] for r in supported) / len(supported),
        'missing_classes': [r['class_id'] for r in per_class if not r['support']],
        'confusion_matrix': matrix, 'per_class': per_class,
    }
