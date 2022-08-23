import copy
import tempfile
import os

import yaml
import pytest

from ..des_files import (
    replace_imsim_data,
    get_pizza_cutter_yaml_path,
    replace_imsim_data_in_pizza_cutter_yaml,
    read_pizza_cutter_yaml,
    PIZZA_CUTTER_YAML_PATH_KEYS,
)


@pytest.mark.parametrize("pth,imsim_data,old_imsim_data,val", [
    (
        "/blah/run/DESr1431/sources-r/OPS/fname",
        "/blah/blah",
        None,
        "/blah/blah/run/DESr1431/sources-r/OPS/fname",
    ),
    (
        "/blah/run/DESr1431/sources-r/OPS/fname",
        "/blah/blah",
        "/blah",
        "/blah/blah/run/DESr1431/sources-r/OPS/fname",
    ),
    (
        "/blah/run/DESr1431/sources-r/ACT/fname",
        "/blah/blah",
        None,
        "/blah/blah/run/DESr1431/sources-r/ACT/fname",
    ),
    (
        "/data/des81.a/data/mtabbutt/Y6_integrations/sim_outputs/"
        "des-pizza-slices-y6-v13/balrog_images/0/fname",
        "/blah/blah",
        None,
        "/blah/blah/"
        "des-pizza-slices-y6-v13/balrog_images/0/fname",
    ),
])
def test_replace_imsim_data(pth, imsim_data, old_imsim_data, val):
    assert val == replace_imsim_data(pth, imsim_data, old_imsim_data=old_imsim_data)


@pytest.mark.parametrize("pth", [
    "/blah/sources-r/dfs",
    "/blah/dfs",
    "/blah/DES/dfs",
])
def test_replace_imsim_data_raises(pth):
    with pytest.raises(RuntimeError):
        replace_imsim_data(pth, "/blah/blah")


def test_get_pizza_cutter_yaml_path():
    pth = get_pizza_cutter_yaml_path("/a", "f", "d", "r")
    assert pth == "/a/f/pizza_cutter_info/d_r_pizza_cutter_info.yaml"


def test_replace_imsim_data_in_pizza_cutter_yaml(pizza_cutter_yaml):
    band_info = copy.deepcopy(pizza_cutter_yaml)

    replace_imsim_data_in_pizza_cutter_yaml(
        band_info, "/blah",
    )
    assert band_info != pizza_cutter_yaml

    for k, v in band_info.items():
        if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
            assert v.startswith("/blah/test-y6-sims/DES")

    for src in band_info["src_info"]:
        for k, v in src.items():
            if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
                assert v.startswith("/blah/test-y6-sims/DES")


def test_pizza_cutter_yaml_io(pizza_cutter_yaml):
    desrun = "rrrrruuuuun"
    tilename = "DES"
    band = "r"
    with tempfile.TemporaryDirectory() as tmpdir:
        imsim_data = os.path.join(tmpdir, "blah")
        pth = get_pizza_cutter_yaml_path(imsim_data, desrun, tilename, band)
        os.makedirs(os.path.dirname(pth), exist_ok=True)

        with open(pth, "w") as fp:
            yaml.dump(pizza_cutter_yaml, fp)

        new_yaml = read_pizza_cutter_yaml(imsim_data, desrun, tilename, band)
        for k, v in new_yaml.items():
            if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
                assert v.startswith(imsim_data)
            elif k != "src_info":
                assert new_yaml[k] == pizza_cutter_yaml[k]

        for i, src in enumerate(new_yaml["src_info"]):
            for k, v in src.items():
                if k.endswith("_path") or k in PIZZA_CUTTER_YAML_PATH_KEYS:
                    assert v.startswith(imsim_data)
                else:
                    assert src[k] == pizza_cutter_yaml["src_info"][i][k]
