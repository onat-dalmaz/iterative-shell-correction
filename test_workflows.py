import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import image_metrics, magnitude_gain, mean_field_bins
from experiments import config, load_array


class WorkflowTests(unittest.TestCase):
    def manifest_config(self, action, selected=None, workflow='volunteers', count=None,
                        cap=60, duplicate_id=False):
        ids = [f'S{i:02}' for i in range(1, 6)] if workflow == 'volunteers' else [
            'standard_autoshim', 'modified_x', 'modified_y']
        if count is not None:
            ids = ids[:count]
        if duplicate_id:
            ids[-1] = ids[0]
        settings = dict(kind=workflow, cases=[dict(id=case) for case in ids],
                        correction=dict(shells=16, shell_spacing_ms=.6, updates=3,
                                        update_cap_hz=cap, final_shells=16))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'inputs.json'
            file.write_text(json.dumps(settings))
            args = argparse.Namespace(inputs=file, workflow=workflow, action=action, case=selected)
            result = config(args)
            self.assertEqual(json.loads(file.read_text()), settings)
        return result

    def test_case_selector_retains_full_default_and_selects_run_or_check(self):
        self.assertEqual(len(self.manifest_config('run')['cases']), 5)
        self.assertEqual(len(self.manifest_config('check', workflow='phantom')['cases']), 3)
        for action in ('run', 'check'):
            with self.subTest(action=action):
                self.assertEqual(self.manifest_config(action, 'S03')['cases'], [dict(id='S03')])
                self.assertEqual(self.manifest_config(action, 'standard_autoshim', 'phantom')['cases'],
                                 [dict(id='standard_autoshim')])

    def test_case_selector_does_not_bypass_full_manifest_validation(self):
        with self.assertRaisesRegex(ValueError, 'requires 5 fixed cases'):
            self.manifest_config('run', 'S01', count=1)
        with self.assertRaisesRegex(ValueError, 'fixed abstract correction settings'):
            self.manifest_config('check', 'S03', cap=120)

    def test_case_selector_is_rejected_for_cohort_scoring(self):
        self.assertEqual(len(self.manifest_config('score')['cases']), 5)
        self.assertEqual(len(self.manifest_config('score', workflow='phantom')['cases']), 3)
        for workflow, selected in [('volunteers', 'S03'), ('phantom', 'standard_autoshim')]:
            with self.subTest(workflow=workflow):
                with self.assertRaisesRegex(ValueError, 'not cohort scoring'):
                    self.manifest_config('score', selected, workflow)

    def test_case_selector_requires_one_exact_id(self):
        with self.assertRaisesRegex(ValueError, 'exactly one configured case'):
            self.manifest_config('run', 'S3')
        with self.assertRaisesRegex(ValueError, 'exactly one configured case'):
            self.manifest_config('check', 'S01', duplicate_id=True)

    def test_signed_volunteer_errors_cancel_before_absolute_value(self):
        volunteers = []
        for value in (-100.0, 50.0, 50.0, 0.0, 0.0):
            volunteers.append([dict(bin_lo_hz=float(low), bin_hi_hz=float(low + 5),
                                    x_hz=low + 2.5, count=60, signed_mean_hz=value)
                               for low in range(0, 200, 5)])
        result = mean_field_bins(volunteers)
        self.assertEqual(len(result), 40)
        self.assertTrue(all(r['absolute_mean_signed_disagreement_hz'] == 0 for r in result))
        self.assertEqual(np.mean(np.abs([-100, 50, 50, 0, 0])), 40)
        volunteers[-1][0]['count'] = 49
        with self.assertRaisesRegex(ValueError, '>=50'):
            mean_field_bins(volunteers)

    def test_gain_samples_nonempty_slices_before_stride(self):
        original = np.full((8, 2, 2), 2 + 1j, np.complex64)
        image = original * np.float32(3)
        mask = np.ones(original.shape, bool)
        mask[1] = False
        gain = magnitude_gain(original, image, mask)
        self.assertEqual(gain['sampled_slices'], [0, 5])
        self.assertAlmostEqual(gain['gain'], 1 / 3, places=7)

    def test_identical_image_metrics(self):
        rng = np.random.default_rng(117)
        reference = rng.uniform(.1, 1, (14, 35, 37))
        mask = np.ones(reference.shape, bool)
        values = image_metrics(reference, reference, mask, (1.5, .546875, .546875), 1)
        self.assertTrue(values['perfect_agreement'])
        self.assertIsNone(values['psnr'])
        self.assertEqual(values['ssim_3d_sigma1p5mm'], 1)
        self.assertEqual(values['hfen_3d_sigma1p5mm'], 0)

    def test_changed_private_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            file = directory / 'mask.npy'
            np.save(file, np.ones((3, 4, 5), bool))
            record = dict(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            self.assertEqual(load_array(record, directory).shape, (3, 4, 5))
            np.save(file, np.zeros((3, 4, 5), bool))
            with self.assertRaisesRegex(ValueError, 'Frozen input differs'):
                load_array(record, directory)


if __name__ == '__main__':
    unittest.main()
