import pytest


class DummyConfig:
    def __init__(self):
        self.option = type('Option', (), {'disable_warnings': False})()

    def addinivalue_line(self, *_):
        pass

    def getoption(self, *_, **__):
        return False


def test_xdist_gpu_binding_reports_no_visible_gpu(monkeypatch):
    import tests.pytest_benchmark_plugin as plugin

    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    monkeypatch.setattr(plugin.torch.cuda, 'device_count', lambda: 0)

    with pytest.raises(pytest.UsageError, match='at least one visible CUDA device'):
        plugin.pytest_configure(DummyConfig())
