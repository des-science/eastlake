import os
import pytest
import tempfile

from ..stash import Stash


TEST_DIR = os.getcwd() + '/'


def test_stash_state():  # test from line 30-35.
    stsh = Stash("blah", ["foo", "bar"])
    assert os.path.abspath("blah") == TEST_DIR + 'blah'
    assert stsh["base_dir"] == os.path.abspath("blah")
    assert stsh["step_names"] == ["foo", "bar"]
    assert stsh['completed_step_names'] == []
    assert stsh['env'] == []
    assert stsh['orig_base_dirs'] == []


def test_stash_state_update():  # test from line 37-51.
    stsh1 = Stash("blah1", ["foo1", "bar1"])  # stash is None
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)  # stash is not None
    assert stsh2["base_dir"] == os.path.abspath("blah2")
    assert stsh2["step_names"] == ["foo1", "bar1"]
    assert stsh2["orig_base_dirs"] == [os.path.abspath("blah1")]  # Why?


def test_stash_get_abs_path():  # test from line 53-60.

    stsh1 = Stash("blah1", ["foo1", "bar1"])
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)
    stsh2['tile_info'] = {
        'tile1': {
            'r': {
                'key1': 'path_to_file1'},
            'i': {},
            'z': {}},
        'tile2': {
            'key2': 'path_to_file2'},
        'tile3': {}}

    # When band==None
    paths = stsh2.get_abs_path('key2', 'tile2', band=None)
    # assert, do we need a test for os.path.isabs() ?
    assert paths == TEST_DIR + 'blah2/path_to_file2'

    # When band!=None
    paths = stsh2.get_abs_path('key1', 'tile1', band='r')
    assert paths == TEST_DIR + 'blah2/path_to_file1'


def test_stash_set_filepaths():  # test from line 62-86.
    stsh1 = Stash("blah1", ["foo1", "bar1"])
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)
    stsh2['tile_info'] = {
        'tile1': {
            'r': {
                'key1': 'path_to_file1'},
            'i': {},
            'z': {}},
        'tile2': {
            'key2': 'path_to_file2'},
        'tile3': {}}

    # when filepaths argument is not list. & band is None.
    # filepaths is absolute.
    stsh2.set_filepaths('key2', TEST_DIR + 'blah2/filepaths2', 'tile2', band=None)
    assert stsh2['tile_info']['tile2']['key2'] == 'filepaths2'
    # filepaths is already relative.
    stsh2.set_filepaths('key2', 'filepaths2', 'tile2', band=None)
    assert stsh2['tile_info']['tile2']['key2'] == 'filepaths2'

    # when filepaths argument is not list. & band is not None.
    # filepaths is absolute.
    stsh2.set_filepaths('key1', TEST_DIR + 'blah2/filepaths1', 'tile1', band='r')
    assert stsh2['tile_info']['tile1']['r']['key1'] == 'filepaths1'
    # filepaths is already relative.
    stsh2.set_filepaths('key1', 'filepaths1', 'tile1', band='r')
    assert stsh2['tile_info']['tile1']['r']['key1'] == 'filepaths1'

    # when filepaths argumet is list. & band is None.
    paths_list = [TEST_DIR + 'blah2/paths1', TEST_DIR + 'blah2/paths2']
    # filepaths is absolute.
    stsh2.set_filepaths('key2', paths_list, 'tile2', band=None)
    assert stsh2['tile_info']['tile2']['key2'] == ['paths1', 'paths2']
    paths_list = ['paths1', 'paths2']
    # filepaths is relative.
    stsh2.set_filepaths('key2', paths_list, 'tile2', band=None)
    assert stsh2['tile_info']['tile2']['key2'] == ['paths1', 'paths2']

    # when filepaths argumet is list. & band is not None.
    paths_list = [TEST_DIR + 'blah2/paths1', TEST_DIR + 'blah2/paths2']
    stsh2.set_filepaths('key1', paths_list, 'tile1', band='r')  # filepaths is absolute.
    assert stsh2['tile_info']['tile1']['r']['key1'] == ['paths1', 'paths2']
    paths_list = ['paths1', 'paths2']
    stsh2.set_filepaths('key1', paths_list, 'tile1', band='r')  # filepaths is relative.
    assert stsh2['tile_info']['tile1']['r']['key1'] == ['paths1', 'paths2']


def test_stash_get_filepaths():  # test from line 88-123.
    stsh1 = Stash("blah1", ["foo1", "bar1"])
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)
    # test for catching exceptions part. (no key exists.)
    for band in [None, 'r']:
        stsh2['tile_info'] = {
            'tile1': {
                'r': {
                    'key1': 'path_to_file1'},
                'i': {},
                'z': {}},
            'tile2': {
                'key2': 'path_to_file2'},
            'tile3': {}}
        # testing of raise(Keyerror)
        with pytest.raises(KeyError) as e:
            # There is no 'key2' for tile1.
            res = stsh2.get_filepaths(
                'key2', 'tile1', band=band, ret_abs=True, keyerror=True)
            assert e.type is KeyError

        # There is no 'key2' for tile1.
        res = stsh2.get_filepaths(
            'key2',
            'tile1',
            band=band,
            ret_abs=True,
            keyerror=False)
        assert res is None

    # test for when there is a key.
    stsh2['tile_info'] = {
        'tile1': {
            'r': {
                'key1': 'path_to_file1'},
            'i': {},
            'z': {}},
        'tile2': {
            'key2': 'path_to_file2'},
        'tile3': {}}
    for band in [None, 'r']:
        if band is None:
            tilename = 'tile2'
            file_key = 'key2'
        elif band == 'r':
            tilename = 'tile1'
            file_key = 'key1'

        arguments = {
            'str': {
                'rel': 'paths1',
                'abs': TEST_DIR + 'blah2/paths1'},
            'list': {
                'rel': [
                    'paths1',
                    'paths2'],
                'abs': [
                    TEST_DIR + 'blah2/paths1',
                    TEST_DIR + 'blah2/paths2']}}
        for i in arguments.keys():
            for j in ['rel', 'abs']:
                if band is None:
                    # change the stsh2['tile_info'][tilename][band][file_key]
                    stsh2['tile_info'][tilename][file_key] = arguments[i][j]
                else:
                    stsh2['tile_info'][tilename][band][file_key] = arguments[i][j]
                # When ret_abs is True.
                res = stsh2.get_filepaths(
                    file_key, tilename, band=band, ret_abs=True, keyerror=True)
                assert res == arguments[i]['abs']  # result is always absolute path.

                # When ret_abs is False. Line 118 in stash.py should be triggered, which
                # returns a relative path, not absolute.
                res = stsh2.get_filepaths(
                    file_key, tilename, band=band, ret_abs=False, keyerror=True)
                assert res == arguments[i][j]


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
            # assert np.array_equal(np.array([1, 2]), np.array([1, 2]))
            # assert np.allclose(np.array([1, 2]), np.array([1, 2]))
            # testing overwrite after the file is made.
            if not ow:
                with pytest.raises(IOError) as e:
                    stsh.save(pth, overwrite=ow)
                    assert e.type is IOError
