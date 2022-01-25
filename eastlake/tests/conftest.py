import yaml

import pytest


@pytest.fixture
def pizza_cutter_yaml():
    return yaml.safe_load("""\
band: r
bmask_ext: msk
bmask_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/coadd/DES0131-3206_r4920p01_r.fits.fz
cat_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/cat/DES0131-3206_r4920p01_r_cat.fits
compression: .fz
crossra0: N
filename: DES0131-3206_r4920p01_r.fits
gaia_stars_file: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/cal/cat_tile_gaia/v1/DES0131-3206_GAIA_DR2_v1.fits
image_ext: sci
image_flags: 0
image_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/coadd/DES0131-3206_r4920p01_r.fits.fz
weight_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/coadd/DES0131-3206_r4920p01_r.fits.fz
weight_ext: wgt
image_shape:
- 10000
- 10000
magzp: 30.0
path: OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/coadd
pfw_attempt_id: 2819137
position_offset: 1
psf_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/psf/DES0131-3206_r4920p01_r_psfcat.psf
scale: 1.0
seg_ext: sci
seg_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/seg/DES0131-3206_r4920p01_r_segmap.fits
src_info:
- band: r
  bkg_ext: sci
  bkg_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/red/bkg/D00492768_r_c11_r4474p01_bkg.fits.fz
  bmask_ext: msk
  bmask_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/red/immask/D00492768_r_c11_r4474p01_immasked.fits.fz
  ccdnum: 11
  compression: .fz
  expnum: 492768
  filename: D00492768_r_c11_r4474p01_immasked.fits
  head_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/aux/DES0131-3206_r4920p01_D00492768_r_c11_scamp.ohead
  image_ext: sci
  image_flags: 0
  image_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/red/immask/D00492768_r_c11_r4474p01_immasked.fits.fz
  image_shape:
  - 4096
  - 2048
  magzp: 31.63902473449707
  path: OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/red/immask
  pfw_attempt_id: 2819137
  piff_info:
    ccdnum: 11
    desdm_flags: 0
    exp_star_t_mean: 0.5134634854806818
    exp_star_t_std: 0.025381696566144734
    expnum: 492768
    fwhm_cen: 1.177111438298789
    nstar: 74
    star_t_mean: 0.4997464380705269
    star_t_std: 0.010614130533410263
  piff_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A2_PIFF/20151113-r5702/D00492768/p01/psf/D00492768_r_c11_r5702p01_piff-model.fits
  position_offset: 1
  psf_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/psf/D00492768_r_c11_r4474p01_psfexcat.psf
  psfex_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/psf/D00492768_r_c11_r4474p01_psfexcat.psf
  scale: 0.22099889703237063
  seg_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/seg/D00492768_r_c11_r4474p01_segmap.fits.fz
  tilename: DES0131-3206
  weight_ext: wgt
  weight_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A1/r4474/20151113/D00492768/p01/red/immask/D00492768_r_c11_r4474p01_immasked.fits.fz
- band: r
  bkg_ext: sci
  bkg_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/red/bkg/D00705196_r_c07_r3518p01_bkg.fits.fz
  bmask_ext: msk
  bmask_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/red/immask/D00705196_r_c07_r3518p01_immasked.fits.fz
  ccdnum: 7
  compression: .fz
  expnum: 705196
  filename: D00705196_r_c07_r3518p01_immasked.fits
  head_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/multiepoch/Y6A1/r4920/DES0131-3206/p01/aux/DES0131-3206_r4920p01_D00705196_r_c07_scamp.ohead
  image_ext: sci
  image_flags: 0
  image_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/red/immask/D00705196_r_c07_r3518p01_immasked.fits.fz
  image_shape:
  - 4096
  - 2048
  magzp: 31.654098510742188
  path: OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/red/immask
  pfw_attempt_id: 2819137
  piff_info:
    ccdnum: 7
    desdm_flags: 0
    exp_star_t_mean: 0.385016429536701
    exp_star_t_std: 0.02182519688157328
    expnum: 705196
    fwhm_cen: 1.0292098344240064
    nstar: 73
    star_t_mean: 0.3820519339122818
    star_t_std: 0.010469793388947353
  piff_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y6A2_PIFF/20171214-r5702/D00705196/p01/psf/D00705196_r_c07_r5702p01_piff-model.fits
  position_offset: 1
  psf_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/psf/D00705196_r_c07_r3518p01_psfexcat.psf
  psfex_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/psf/D00705196_r_c07_r3518p01_psfexcat.psf
  scale: 0.2179518680631856
  seg_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/seg/D00705196_r_c07_r3518p01_segmap.fits.fz
  tilename: DES0131-3206
  weight_ext: wgt
  weight_path: /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/finalcut/Y5A1/r3518/20171214/D00705196/p01/red/immask/D00705196_r_c07_r3518p01_immasked.fits.fz
""")  # noqa
