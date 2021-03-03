import os

from ..stash import Stash


def test_stash_state():
    stsh = Stash("blah", ["foo", "bar"])
    assert stsh["base_dir"] == os.path.abspath("blah")
    assert stsh["step_names"] == ["foo", "bar"]


def test_stash_state_update():
    stsh1 = Stash("blah1", ["foo1", "bar1"])
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)
    assert stsh2["base_dir"] == os.path.abspath("blah2")
    assert stsh2["step_names"] == ["foo1", "bar1"]
    assert stsh2["orig_base_dirs"] == [os.path.abspath("blah1")]
