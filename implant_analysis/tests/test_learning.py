"""Opt-in, actual CPU ResNet50 smoke tests. No medical files or downloads."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from implant_analysis.common import file_sha256, load_config, write_csv, write_json
from implant_analysis.study import freeze_split, load_study


@unittest.skipUnless(os.environ.get('IMPLANT_RUN_ML_TESTS') == '1',
                     'Set IMPLANT_RUN_ML_TESTS=1 with the approved ML dependencies installed')
class LearningTests(unittest.TestCase):
    def test_train_resume_evaluate_and_safety_boundaries(self):
        import torch
        from torchvision.models import resnet50
        from implant_analysis.learning import train, evaluate
        from implant_analysis.figures import plot_results
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'data/images'
            images.mkdir(parents=True)
            config = dict(data_root=str(root / 'data'), image_root='images', label_root='labels',
                work_dir=str(root / 'work'), pairing={'verified': True},
                grouping={'verified': True, 'kind': 'patient'},
                training=dict(device='cpu', epochs=1, warmup_epochs=0, image_size=32,
                              batch_size=2, amp=False, patience=5))
            write_json(root / 'site.json', config)
            cfg = load_config(root / 'site.json')
            rows = []
            for partition in ('train', 'dev', 'calibration', 'test'):
                for k in range(2):
                    name = f'{partition}{k}'
                    path = images / (name + '.png')
                    Image.new('L', (20, 32), 40 + len(rows) * 20).save(path)
                    rows.append(dict(sample_id=name, split=partition, class_id=str(k),
                        class_key=f'manufacturer/model{k}', group_id=name, source_image_group_id='',
                        leakage_group_id=name, image_sha256=file_sha256(path), pixel_sha256=name,
                        image_relpath=path.name, status='eligible', hospital='synthetic', modality='synthetic'))
            draft = root / 'draft'
            draft.mkdir()
            write_csv(draft / 'split_manifest.csv', rows, list(rows[0]))
            write_json(draft / 'config_snapshot.json', config)
            write_json(draft / 'summary.json', dict(status='DRAFT_REQUIRES_REVIEW_BEFORE_TRAINING',
                split_manifest_sha256=file_sha256(draft / 'split_manifest.csv'), grouping_kind='patient'))
            frozen = freeze_split(cfg, draft, 'synthetic', 'Generated synthetic groups only')
            with self.assertRaisesRegex(ValueError, '가중치'):
                train(cfg, frozen, 'missing_weights')
            cfg['training']['weights_path'] = str(root / 'bad.pth')
            cfg['training']['weights_sha256'] = '0' * 64
            (root / 'bad.pth').write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                train(cfg, frozen, 'wrong_hash')
            # Held-out files are inaccessible: training must still succeed.
            for row in rows:
                if row['split'] in ('test', 'calibration'):
                    (images / row['image_relpath']).rename(images / (row['image_relpath'] + '.hidden'))
            with patch('torch.hub.download_url_to_file', side_effect=AssertionError('Network forbidden')):
                # Exercise local 1000-class state_dict loading and classifier replacement.
                weights = root / 'synthetic_resnet50.pth'
                torch.save(resnet50(weights=None).state_dict(), weights)
                cfg['training']['weights_path'] = str(weights)
                cfg['training']['weights_sha256'] = file_sha256(weights)
                local_weights_run = train(cfg, frozen, 'local_weights')
                self.assertTrue((local_weights_run / 'last.pt').is_file())
                first = train(cfg, frozen, 'epoch1', synthetic_smoke=True)
                cfg['training']['epochs'] = 2
                resumed = train(cfg, frozen, 'resumed', first / 'last.pt', synthetic_smoke=True)
                straight = train(cfg, frozen, 'straight', synthetic_smoke=True)
            a = torch.load(resumed / 'last.pt', weights_only=True, map_location='cpu')
            b = torch.load(straight / 'last.pt', weights_only=True, map_location='cpu')
            self.assertEqual(a['history'], b['history'])
            self.assertTrue(all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']))
            del a, b
            with self.assertRaisesRegex(ValueError, 'final-test-note'):
                evaluate(cfg, frozen, resumed / 'last.pt', 'test', 'no_test_permission')
            result = evaluate(cfg, frozen, resumed / 'last.pt', 'dev', 'dev_only')
            metrics = json.loads((result / 'metrics.json').read_text())
            self.assertEqual(metrics['n'], 2)
            self.assertTrue(metrics['synthetic_smoke'])
            figures = plot_results(result, resumed)
            self.assertTrue((figures / 'confusion_matrix.png').is_file())
            self.assertTrue((figures / 'learning_curves.pdf').is_file())
            with self.assertRaisesRegex(ValueError, '체크포인트가 다릅니다'):
                plot_results(result, first)
            for row in rows:
                if row['split'] == 'test':
                    (images / (row['image_relpath'] + '.hidden')).rename(images / row['image_relpath'])
            final = evaluate(cfg, frozen, resumed / 'last.pt', 'test', 'synthetic_final',
                             'Synthetic test only; fixed settings')
            self.assertEqual(json.loads((final / 'metrics.json').read_text())['partition'], 'test')
            # Mutation after audit is detected before any model fit.
            (images / rows[0]['image_relpath']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '영상이 변경'):
                train(cfg, frozen, 'changed', synthetic_smoke=True)
            (frozen / 'split_manifest.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, '변경'):
                load_study(frozen)


if __name__ == '__main__':
    unittest.main()
