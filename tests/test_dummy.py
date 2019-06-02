def test_local():
    from pycsmaca import dummy
    assert dummy.welcome() == 'Welcome from pycsmaca'


def test_pydesim_dep():
    from pydesim import dummy
    assert dummy.hello() == 'Hello World'
