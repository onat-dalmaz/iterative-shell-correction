#!/bin/sh
# Self-tests. ring_b0_corr's regression test needs the original translation
# (cones_recon_fcns.py) on sys.path; run from the parent ring_corr/ checkout,
# or skip that line. time_seg_corr's test needs brian_ref/tsphase/*.mat.
set -e
python ring_b0_corr.py --selftest
python time_seg_corr.py --selftest
python autofocus_blend.py --selftest
