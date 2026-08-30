"""Offline ResNet50 baseline. Imported only for train/evaluate commands."""
import collections
import json
import math
import os
import platform
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

from .common import config_path, file_sha256, new_output, token, write_csv, write_json
from .metrics import classification_metrics
from .study import image_path, load_study, verify_images


DEFAULTS = dict(epochs=10, warmup_epochs=2, batch_size=16, image_size=224,
                head_lr=0.001, backbone_lr=0.0001, weight_decay=0.0001,
                patience=5, device='cuda', seed=42, rotation_degrees=3.0,
                brightness_contrast=0.1, amp=True)


class CropDataset(Dataset):
    def __init__(self, study, rows, options, augment=False):
        self.study, self.rows, self.size = study, rows, options['image_size']
        operations = []
        if augment:
            operations.extend([
                transforms.RandomRotation(options['rotation_degrees'], fill=0),
                transforms.ColorJitter(brightness=options['brightness_contrast'],
                                       contrast=options['brightness_contrast'])])
        operations.extend([transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(self.study, row)) as image:
            # Keep the entire implant and its aspect ratio; no random/central crop.
            image = ImageOps.pad(image.convert('RGB'), (self.size, self.size),
                                 method=Image.Resampling.BILINEAR, color=0)
        return self.transform(image), int(row['class_id']), index


def options_from(config):
    supplied = config.get('training', {})
    allowed = set(DEFAULTS) | {'weights_path', 'weights_sha256'}
    if set(supplied) - allowed:
        raise ValueError(f'알 수 없는 학습 설정: {sorted(set(supplied) - allowed)}')
    options = {**DEFAULTS, **{k: v for k, v in supplied.items() if k in DEFAULTS}}
    for key in ('epochs', 'batch_size', 'image_size', 'patience'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'{key}는 양의 정수여야 합니다.')
    if options['image_size'] < 32 or type(options['warmup_epochs']) is not int or options['warmup_epochs'] < 0:
        raise ValueError('image_size는 32 이상, warmup_epochs는 0 이상이어야 합니다.')
    if type(options['seed']) is not int:
        raise ValueError('seed는 정수여야 합니다.')
    for key in ('head_lr', 'backbone_lr', 'weight_decay', 'rotation_degrees', 'brightness_contrast'):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'{key} 값을 확인하세요.')
    if options['head_lr'] == 0 or options['backbone_lr'] == 0:
        raise ValueError('학습률은 0보다 커야 합니다.')
    if type(options['amp']) is not bool:
        raise ValueError('amp는 JSON의 true 또는 false여야 합니다.')
    if options['device'] not in ('cpu', 'cuda'):
        raise ValueError('device는 cpu 또는 cuda로 지정하세요.')
    if options['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA를 사용할 수 없습니다. 설치·GPU를 확인하거나 cpu를 명시하세요.')
    return options


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_model(class_count, weights=None):
    model = resnet50(weights=None)  # Never let torchvision download anything.
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(('layer4.', 'fc.'))
    return model


def loader(study, rows, options, training=False):
    return DataLoader(CropDataset(study, rows, options, training),
                      batch_size=options['batch_size'], shuffle=training, num_workers=0,
                      pin_memory=options['device'] == 'cuda')


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    targets, probabilities, indexes = [], [], []
    for images, labels, index in batches:
        logits = model(images.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError('유한하지 않은 예측값입니다.')
        targets.extend(labels.tolist())
        probabilities.extend(logits.softmax(1).cpu().tolist())
        indexes.extend(index.tolist())
    return targets, probabilities, indexes


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def check_data_config(config, study):
    if str(config_path(config, config['data_root'])) != study['data_root'] or config['image_root'] != study['image_root']:
        raise ValueError('확정 분할과 현재 원본 데이터 경로가 다릅니다.')


def train(config, frozen_dir, run_id, resume=None, synthetic_smoke=False):
    study, rows, classes = load_study(frozen_dir)
    check_data_config(config, study)
    options = options_from(config)
    seed_everything(options['seed'])
    selected = {p: [r for r in rows if r['split'] == p] for p in ('train', 'dev')}
    # No calibration/test image is opened here, including in the preflight.
    verify_images(study, selected['train'] + selected['dev'])
    previous = None
    source_hash = None
    if resume:
        previous = torch.load(Path(resume), map_location='cpu', weights_only=True)
        if previous['study_sha256'] != study['study_sha256'] or previous['classes'] != classes:
            raise ValueError('재개 체크포인트와 확정 분할이 다릅니다.')
        comparable = lambda o: {k: v for k, v in o.items() if k != 'epochs'}
        if comparable(previous['options']) != comparable(options) or previous['synthetic_smoke'] != synthetic_smoke:
            raise ValueError('재개 시 epochs 외 학습 설정·합성 시험 모드를 변경할 수 없습니다.')
        if previous['epoch'] >= options['epochs'] or previous['stopped_early']:
            raise ValueError('이미 끝난 학습입니다. 조기 종료 모델을 재개해 dev 선택을 반복하지 마세요.')
        source_hash = previous['pretrained_sha256']
    weights = None
    if not previous and not synthetic_smoke:
        settings = config.get('training', {})
        if not settings.get('weights_path') or not settings.get('weights_sha256'):
            raise ValueError('승인된 로컬 가중치 경로와 전체 SHA-256을 지정하세요.')
        weights = config_path(config, settings['weights_path'])
        source_hash = file_sha256(weights)
        if source_hash != settings['weights_sha256'].lower():
            raise ValueError('사전학습 가중치 SHA-256이 일치하지 않습니다.')
    model = make_model(len(classes), weights).to(options['device'])
    optimizer = torch.optim.AdamW([
        {'params': model.fc.parameters(), 'lr': options['head_lr']},
        {'params': model.layer4.parameters(), 'lr': options['backbone_lr']}],
        weight_decay=options['weight_decay'])
    use_amp = bool(options['amp'] and options['device'] == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    start, history, best_score, best_epoch, stale, best_state = 0, [], -1.0, 0, 0, None
    if previous:
        model.load_state_dict(previous['model'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        start, history = previous['epoch'], previous['history']
        best_score, best_epoch = previous['best_score'], previous['best_epoch']
        stale, best_state = previous['stale'], previous['best_model']
        del previous
    out = new_output(config, 'train', run_id)
    write_json(out / 'environment.json', dict(python=platform.python_version(), torch=str(torch.__version__),
        torchvision=str(torchvision.__version__), device=options['device'], cuda=torch.version.cuda,
        study_sha256=study['study_sha256'], synthetic_smoke=synthetic_smoke,
        source_sha256=token([(p.name, file_sha256(p)) for p in sorted(Path(__file__).parent.glob('*.py'))]),
        resume_sha256=file_sha256(resume) if resume else None))
    dev_loader = loader(study, selected['dev'], options)
    for epoch in range(start, options['epochs']):
        # Epoch-boundary restart: seed/order/augmentations are independent of previous epochs.
        seed_everything(options['seed'] + epoch)
        model.eval()  # Keep all BatchNorm running statistics fixed, including layer4.
        for parameter in model.layer4.parameters():
            parameter.requires_grad = epoch >= options['warmup_epochs']
        loss_total, count = 0.0, 0
        for images, labels, _ in loader(study, selected['train'], options, training=True):
            images, labels = images.to(options['device']), labels.to(options['device'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=options['device'], enabled=use_amp):
                loss = nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError('학습 loss가 유한하지 않습니다. 직전 완료 epoch에서 재개하세요.')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item() * len(labels)
            count += len(labels)
        targets, probs, _ = predict(model, dev_loader, options['device'])
        metrics = classification_metrics(targets, probs, len(classes))
        score = metrics['macro_f1_all_classes']
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append(dict(epoch=epoch + 1, train_loss=loss_total / count,
                            dev_macro_f1=score, dev_accuracy=metrics['accuracy']))
        checkpoint = dict(format_version=1, architecture='resnet50', epoch=epoch + 1,
            model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            best_model=best_state, best_score=best_score, best_epoch=best_epoch, stale=stale,
            stopped_early=stale >= options['patience'], options=options, history=history,
            classes=classes, study_sha256=study['study_sha256'], pretrained_sha256=source_hash,
            synthetic_smoke=synthetic_smoke)
        atomic_checkpoint(out / 'last.pt', checkpoint)
        write_json(out / 'history.json', history)
        print(f"Epoch {epoch + 1}/{options['epochs']} loss={loss_total/count:.4f} dev macro-F1={score:.4f}", flush=True)
        if checkpoint['stopped_early']:
            break
    return out


def evaluate(config, frozen_dir, checkpoint_path, partition, run_id, final_test_note=None):
    if partition not in ('dev', 'test'):
        raise ValueError('현재 기준모델 평가는 dev 또는 test만 지원합니다. calibration은 보존합니다.')
    if partition == 'test' and (not final_test_note or not final_test_note.strip()):
        raise ValueError('모델·전처리·평가기준 확정을 기록하는 final-test-note가 필요합니다.')
    study, rows, classes = load_study(frozen_dir)
    check_data_config(config, study)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['study_sha256'] != study['study_sha256'] or checkpoint['classes'] != classes:
        raise ValueError('체크포인트와 확정 분할이 다릅니다.')
    options = dict(checkpoint['options'])
    # Inference may run on CPU even if training ran on CUDA; preprocessing is fixed.
    options['device'] = config.get('training', {}).get('device', options['device'])
    if options['device'] not in ('cpu', 'cuda') or options['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('평가 device를 확인하세요.')
    selected = [r for r in rows if r['split'] == partition]
    verify_images(study, selected)
    model = make_model(len(classes)).to(options['device'])
    model.load_state_dict(checkpoint['best_model'], strict=True)
    targets, probs, indexes = predict(model, loader(study, selected, options), options['device'])
    metrics = classification_metrics(targets, probs, len(classes))
    out = new_output(config, 'evaluate', run_id)
    write_json(out / 'metrics.json', dict(partition=partition, best_epoch=checkpoint['best_epoch'],
        checkpoint_sha256=file_sha256(checkpoint_path), study_sha256=study['study_sha256'],
        final_test_note=final_test_note, synthetic_smoke=checkpoint['synthetic_smoke'],
        all_outputs_internal_only=True, **metrics))
    predictions = []
    for target, probability, index in zip(targets, probs, indexes):
        row = selected[index]
        predictions.append(dict(sample_id=row['sample_id'], group_id=row['leakage_group_id'],
            hospital=row['hospital'], modality=row['modality'], class_id=target,
            predicted_class=max(range(len(classes)), key=lambda k: probability[k]),
            **{f'p_{k}': p for k, p in enumerate(probability)}))
    write_csv(out / 'predictions.csv', predictions, list(predictions[0]))
    write_json(out / 'class_map.json', classes)
    subgroups = []
    for field in ('hospital', 'modality'):
        groups = collections.defaultdict(list)
        for i, row in enumerate(predictions):
            groups[row[field] or '(missing)'].append(i)
        for value, ids in sorted(groups.items()):
            subgroups.append(dict(field=field, value=value,
                independent_groups=len({predictions[i]['group_id'] for i in ids}),
                **classification_metrics([targets[i] for i in ids], [probs[i] for i in ids], len(classes))))
    write_json(out / 'subgroups.json', subgroups)
    return out
