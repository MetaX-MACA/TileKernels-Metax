import os


class DummyConfig:
    def __init__(self):
        self.option = type('Option', (), {'disable_warnings': False})()

    def addinivalue_line(self, *_):
        pass

    def getoption(self, *_, **__):
        return False


def test_xdist_gpu_binding_skips_without_gpu(monkeypatch):
    import tests.pytest_benchmark_plugin as plugin

    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    monkeypatch.setattr(plugin.torch.cuda, 'device_count', lambda: 0)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)

    plugin.pytest_configure(DummyConfig())
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ
