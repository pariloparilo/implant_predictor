import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from implant_analysis.audit import select_sample
from implant_analysis.common import file_sha256, load_config, read_csv
from implant_analysis.visit import visit


class VisitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        template = Path(__file__).parents[1] / 'configs/visit.example.json'
        self.config = json.loads(template.read_text())
        self.config.update(data_root=str(self.root/'data'), work_dir=str(self.root/'work'))
        self.config['visit']['sample_per_partition'] = 2
        for part in self.config['provider_partitions']:
            for key in ('label_root', 'image_root'):
                (self.root/'data'/part[key]).mkdir(parents=True)
        self.before = {}

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self):
        path = self.root/'visit.json'
        path.write_text(json.dumps(self.config))
        return load_config(path)

    def add(self, part_index, folder, name, color, model='Product'):
        part = self.config['provider_partitions'][part_index]
        rel = Path(folder)/'h01_01'/name
        label = self.root/'data'/part['label_root']/rel.with_suffix('.json')
        image = self.root/'data'/part['image_root']/rel.with_suffix('.jpeg')
        label.parent.mkdir(parents=True, exist_ok=True)
        image.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(json.dumps({'일련번호':'532','제조사':'Synthetic','모델명':model,
                                     '촬영방법':'파노라마','촬영병원':'Synthetic Hospital'}))
        Image.new('L',(30,60),color).save(image)
        self.before[label] = file_sha256(label)
        self.before[image] = file_sha256(image)
        return label, image

    def test_full_preserves_provider_identity_and_detects_cross_duplicates(self):
        self.add(0,'h01','same_name',20)
        self.add(1,'h01','same_name',20,model='Conflicting Product')
        self.add(1,'h02','unique',90)
        out = visit(self.cfg(),'full','full')
        summary = json.loads((out/'summary.json').read_text())
        self.assertEqual(summary['status'],'COMPLETED')
        self.assertFalse(summary['creates_training_split'])
        cross = summary['cross_partition']
        self.assertEqual(cross['scope'],'FULL')
        self.assertEqual(cross['cross_partition_exact_pixel_clusters'],1)
        self.assertEqual(cross['cross_partition_label_conflict_clusters'],1)
        rows = read_csv(out/'provider_manifest_internal.csv')
        self.assertEqual(len({r['qualified_sample_id'] for r in rows}),3)
        self.assertNotIn('split',rows[0])
        self.assertFalse((out/'contact_sheet.png').exists())
        self.assertTrue(all(file_sha256(path)==digest for path,digest in self.before.items()))

    def test_sample_covers_source_folders_and_marks_partial(self):
        for part in range(2):
            for i in range(4):
                self.add(part,'h01',str(i),i*15)
            self.add(part,'h15','b',100)
        out = visit(self.cfg(),'sample','sample')
        rows=read_csv(out/'provider_manifest_internal.csv')
        self.assertEqual(len(rows),4)
        for name in ('provided_train','provided_validation'):
            self.assertEqual({Path(r['label_relpath']).parts[0] for r in rows if r['provider_partition']==name}, {'h01','h15'})
        summary=json.loads((out/'summary.json').read_text())
        self.assertEqual(summary['cross_partition']['scope'],'PARTIAL_OBSERVED_ONLY')
        self.assertTrue((out/'contact_sheet.png').exists())
        with Image.open(out/'contact_sheet.png') as img:
            self.assertEqual(img.width,1000)
        self.assertIn('부분 점검', (out/'READ_ON_SCREEN.txt').read_text())

    def test_preflight_does_not_open_source_contents(self):
        for part in range(2):
            label,image=self.add(part,'h01','a',40)
            label.write_text('invalid json')
            image.write_bytes(b'not an image')
        out=visit(self.cfg(),'pre','preflight')
        summary=json.loads((out/'summary.json').read_text())
        self.assertEqual(summary['status'],'COMPLETED')
        self.assertNotIn('cross_partition',summary)
        self.assertNotIn('audit',summary['partitions']['provided_train'])

    def test_missing_partition_is_partial_error_not_false_success(self):
        self.add(0,'h01','a',40)
        out=visit(self.cfg(),'empty','full')
        summary=json.loads((out/'summary.json').read_text())
        self.assertEqual(summary['status'],'COMPLETED_WITH_ERRORS')
        self.assertEqual(summary['cross_partition']['scope'],'PARTIAL_OBSERVED_ONLY')
        self.assertTrue((out/'dashboard.html').exists())

    def test_path_overlap_and_traversal_rejected(self):
        for path in ('../outside','1.Training'):
            self.config['provider_partitions'][1]['image_root']=path
            with self.assertRaises(ValueError):
                visit(self.cfg(),'no','preflight')

    def test_repeat_run_does_not_overwrite(self):
        self.add(0,'h01','a',40)
        self.add(1,'h01','a',50)
        visit(self.cfg(),'same','preflight')
        with self.assertRaises(FileExistsError):
            visit(self.cfg(),'same','preflight')

    def test_balanced_sampling_is_reproducible(self):
        root=Path('/synthetic')
        paths=[root/f'h{i:02}'/f'{k}.json' for i in (1,2,15) for k in range(10)]
        a=select_sample(paths,root,9,'balanced_folder',42)
        self.assertEqual(a,select_sample(paths,root,9,'balanced_folder',42))
        self.assertEqual({p.parent.name for p in a},{'h01','h02','h15'})
        self.assertEqual(len(set(a)),9)


if __name__=='__main__':
    unittest.main()
