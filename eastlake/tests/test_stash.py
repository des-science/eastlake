from ..stash import Stash


def test_stash_state():
    stsh = Stash("blah", ["foo", "bar"])
    assert stsh["base_dir"] == "blah"
    assert stsg["step_names"] == ["foo", "bar"]
    

def test_stash_state_update():
    stsh1 = Stash("blah1", ["foo1", "bar1"])
    stsh2 = Stash("blah2", ["foo2", "bar2"], stash=stsh1)
    assert stsh["base_dir"] == "blah1"
    assert stsg["step_names"] == ["foo1", "bar1"]
  
