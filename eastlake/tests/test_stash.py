import os
import pytest
import tempfile

from ..stash import Stash
from ..des_files import PIZZA_CUTTER_YAML_PATH_KEYS


TEST_DIR = os.getcwd() + '/'


def test_stash_state(monkeypatch):
    monkeypatch.setenv("IMSIM_DATA", "blahblah")
    stsh = Stash("blah", ["foo", "bar"])
    assert os.path.abspath("blah") == TEST_DIR + 'blah'
    assert stsh["base_dir"] == os.path.abspath("blah")
    assert stsh["step_names"] == ["foo", "bar"]
    assert stsh['completed_step_names'] == []
    assert stsh['env'] == []
    assert stsh['orig_base_dirs'] == []
    assert stsh["imsim_data"] == "blahblah"


def test_stash_state_update():
    stsh1 = Stash("blah1", ["foo1", "bar1"])  # stash is None
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)  # stash is not None
    assert stsh2["base_dir"] == os.path.abspath("blah2")
    assert stsh2["step_names"] == ["foo1", "bar1"]
    assert stsh2["orig_base_dirs"] == [os.path.abspath("blah1")]  # Why?


@pytest.mark.parametrize("in_fnames,out_fnames", [
    (TEST_DIR + 'blah2/filepaths2', "filepaths2"),
    ('filepaths2', "filepaths2"),
    ([TEST_DIR + 'blah2/paths1', TEST_DIR + 'blah2/paths2'], ["paths1", "paths2"]),
    (["paths1", "paths2"], ["paths1", "paths2"]),
])
@pytest.mark.parametrize("ext", [1, None, "sci"])
@pytest.mark.parametrize("band", [None, "r"])
def test_stash_filepaths(band, ext, in_fnames, out_fnames):
    stsh1 = Stash("blah1", ["foo1", "bar1"])
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)

    with pytest.raises(KeyError) as e:
        res = stsh2.get_filepaths(
            'key2', 'tile1', band=band, ret_abs=True, keyerror=True)
        assert e.type is KeyError

    res = stsh2.get_filepaths(
        'key2',
        'tile1',
        band=band,
        ret_abs=True,
        keyerror=False)
    assert res is None

    stsh2.set_filepaths('key2', in_fnames, 'tile2', band=band, ext=ext)
    assert stsh2.get_filepaths('key2', 'tile2', band=band, ret_abs=False) == out_fnames
    if isinstance(out_fnames, str):
        abs_out_fnames = TEST_DIR + "blah2/" + out_fnames
    else:
        abs_out_fnames = [
            TEST_DIR + "blah2/" + ot
            for ot in out_fnames
        ]
    assert stsh2.get_filepaths('key2', 'tile2', band=band, ret_abs=True) == abs_out_fnames

    if ext is not None:
        assert stsh2.get_filepaths(
            'key2', 'tile2', band=band,
            ret_abs=False, with_fits_ext=True
        ) == (out_fnames, ext)

    if band is None:
        assert stsh2['tile_info']['tile2']['key2'] == out_fnames
        if ext is not None:
            assert stsh2['tile_info']['tile2']['key2_ext'] == ext
        else:
            assert "key2_ext" not in stsh2['tile_info']['tile2']
    else:
        assert stsh2['tile_info']['tile2'][band]['key2'] == out_fnames
        if ext is not None:
            assert stsh2['tile_info']['tile2'][band]['key2_ext'] == ext
        else:
            assert "key2_ext" not in stsh2['tile_info']['tile2'][band]


@pytest.mark.parametrize("tilename,band", [("tt", None), ("tt", "g")])
def test_stash_info_quantity(tilename, band):
    key = "blah"
    value = 10
    stsh = Stash("blah1", ["foo1", "bar1"])

    assert not stsh.has_tile_info_quantity(key, tilename, band=band)

    stsh.set_tile_info_quantity(key, value, tilename, band=band)
    assert stsh.has_tile_info_quantity(key, tilename, band=band)
    assert not stsh.has_tile_info_quantity(key+"575", tilename, band=band)

    assert stsh.get_tile_info_quantity(key, tilename, band=band) == value

    with pytest.raises(KeyError):
        stsh.get_tile_info_quantity(key+"575", tilename, band=band)

    assert stsh.get_tile_info_quantity(key+"575", tilename, band=band, keyerror=False) is None

    if band is None:
        with pytest.raises(AssertionError):
            stsh.set_tile_info_quantity("r", value, tilename, band=band)


def test_stash_io():
    stsh = Stash("blah1", ["foo1", "bar1"])
    with tempfile.TemporaryDirectory() as tmpdir:
        pth = os.path.join(tmpdir, "blah.pkl")
        for ow in [False, True]:
            # write stash to disk
            stsh.save(pth, overwrite=ow)
            # read stash back
            loaded_stsh = stsh.load(pth, TEST_DIR + 'blah1', ["foo1", "bar1"])
            # check that they are equal
            assert loaded_stsh == stsh
            # testing overwrite after the file is made.
            if not ow:
                with pytest.raises(IOError) as e:
                    stsh.save(pth, overwrite=ow)
                    assert e.type is IOError


def test_stash_io_pizza_cutter_yaml(pizza_cutter_yaml):
    stsh = Stash("blah1", ["foo1", "bar1"])
    stsh["imsim_data"] = "/imsim_data"
    stsh["desrun"] = "deeeesssssruuuuunnnnn"
    tilename = "ddd"
    band = "g"

    # raises if not there
    with pytest.raises(RuntimeError):
        stsh.get_input_pizza_cutter_yaml(tilename, band)

    with pytest.raises(RuntimeError):
        stsh.get_output_pizza_cutter_yaml(tilename, band)

    # if no output, set a copy on input
    assert not stsh.has_output_pizza_cutter_yaml(tilename, band)
    stsh.set_input_pizza_cutter_yaml(pizza_cutter_yaml, tilename, band)
    assert stsh.has_output_pizza_cutter_yaml(tilename, band)

    # the set copy has imsim_data -> base_dir
    # and the input imsim_data replace with the current one
    new_yaml = stsh.get_input_pizza_cutter_yaml(tilename, band)
    for k, v in new_yaml.items():
        if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
            assert v.startswith(stsh["imsim_data"])
        elif k != "src_info":
            assert new_yaml[k] == pizza_cutter_yaml[k]

    for i, src in enumerate(new_yaml["src_info"]):
        for k, v in src.items():
            if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
                assert v.startswith(stsh["imsim_data"])
            else:
                assert src[k] == pizza_cutter_yaml["src_info"][i][k]

    new_yaml = stsh.get_output_pizza_cutter_yaml(tilename, band)
    for k, v in new_yaml.items():
        if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
            assert v.startswith(stsh["base_dir"])
        elif k != "src_info":
            assert new_yaml[k] == pizza_cutter_yaml[k]

    for i, src in enumerate(new_yaml["src_info"]):
        for k, v in src.items():
            if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
                assert v.startswith(stsh["base_dir"])
            else:
                assert src[k] == pizza_cutter_yaml["src_info"][i][k]

    # now test an update
    with stsh.update_output_pizza_cutter_yaml(tilename, band) as pyml:
        pyml["image_ext"] = 10
    new_yaml = stsh.get_output_pizza_cutter_yaml(tilename, band)
    assert new_yaml["image_ext"] == 10

    # if we already have an output, setting an input should not change it
    stsh.set_input_pizza_cutter_yaml(pizza_cutter_yaml, tilename, band)
    new_yaml = stsh.get_output_pizza_cutter_yaml(tilename, band)
    assert new_yaml["image_ext"] == 10


def test_set_output_pizza_cutter_yaml_tile_info(pizza_cutter_yaml):
    stsh = Stash("blah1", ["foo1", "bar1"])
    stsh["imsim_data"] = "/imsim_data"
    tilename = "ddd"
    band = "g"

    stsh.set_input_pizza_cutter_yaml(pizza_cutter_yaml, tilename, band)
    pyml = stsh.get_output_pizza_cutter_yaml(tilename, band)

    for stsh_key, py_key in [
        ("img", "image"),
        ("wgt", "weight"),
        ("msk", "bmask"),
        ("bkg", "bkg"),
        ("piff", "piff"),
        ("psfex", "psfex"),
    ]:
        for with_fits_ext in [True, False]:
            if py_key in ["piff", "psfex"]:
                with_fits_ext = False

            res = stsh.get_filepaths(f"{stsh_key}_files", tilename, band=band, with_fits_ext=with_fits_ext)
            if with_fits_ext:
                fnames = res[0]
                ext = res[1]
                assert ext == list(set([src[f"{py_key}_ext"] for src in pyml["src_info"]]))[0]
            else:
                fnames = res
                ext = None

            assert fnames == [src[f"{py_key}_path"] for src in pyml["src_info"]]

    mag_zps = stsh.get_tile_info_quantity("mag_zps", tilename, band=band)
    assert mag_zps == [src["magzp"] for src in pyml["src_info"]]

    assert stsh.get_filepaths("srcex_cat", tilename, band=band) == pyml["cat_path"]

    for stsh_key, py_key in [
        ("coadd", "image"),
        ("coadd_weight", "weight"),
        ("coadd_mask", "bmask"),
        ("seg", "seg"),
    ]:
        for with_fits_ext in [True, False]:
            res = stsh.get_filepaths(f"{stsh_key}_file", tilename, band=band, with_fits_ext=with_fits_ext)
            if with_fits_ext:
                fnames = res[0]
                ext = res[1]
                assert ext == pyml[f"{py_key}_ext"]
            else:
                fnames = res
                ext = None

            assert fnames == pyml[f"{py_key}_path"]
