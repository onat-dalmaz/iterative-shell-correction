import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import image_metrics, magnitude_gain, mean_field_bins
from experiments import load_array


class WorkflowTests(unittest.TestCase):
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
