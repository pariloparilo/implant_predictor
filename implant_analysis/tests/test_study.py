import unittest

from implant_analysis.metrics import classification_metrics
from implant_analysis.study import validate_rows


def valid_rows():
    return [dict(sample_id=f'{p}{c}', split=p, status='eligible', class_id=str(c),
                 class_key=f'product{c}', group_id=f'{p}{c}', source_image_group_id='',
                 pixel_sha256=f'pixel{p}{c}', leakage_group_id=f'{p}{c}',
                 image_sha256=f'hash{p}{c}', image_relpath=f'{p}{c}.png')
            for p in ('train', 'dev', 'calibration', 'test') for c in range(2)]


class StudyTests(unittest.TestCase):
    def test_cross_partition_group_rejected(self):
        rows = valid_rows()
        rows[2]['group_id'] = rows[0]['group_id']
        with self.assertRaisesRegex(ValueError, '누수'):
            validate_rows(rows)

    def test_missing_dev_class_rejected(self):
        rows = [r for r in valid_rows() if not (r['split'] == 'dev' and r['class_id'] == '1')]
        with self.assertRaisesRegex(ValueError, 'dev'):
            validate_rows(rows)

    def test_class_map_mismatch_rejected(self):
        rows = valid_rows()
        rows[0]['class_key'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'class_id'):
            validate_rows(rows)

    def test_metrics_known_confusion_and_missing_class(self):
        result = classification_metrics([0, 0, 1], [[.8,.1,.1],[.1,.8,.1],[.1,.8,.1]], 3)
        self.assertEqual(result['confusion_matrix'], [[1,1,0],[0,1,0],[0,0,0]])
        self.assertAlmostEqual(result['accuracy'], 2/3)
        self.assertAlmostEqual(result['macro_f1_all_classes'], 4/9)
        self.assertEqual(result['missing_classes'], [2])
        self.assertIsNone(result['per_class'][2]['recall'])

    def test_nonfinite_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [[float('nan'),1]], 2)


if __name__ == '__main__':
    unittest.main()
