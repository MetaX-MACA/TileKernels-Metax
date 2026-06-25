def test_collect_environment_metadata_without_cuda(monkeypatch):
    import tests.pytest_benchmark_plugin as plugin

    monkeypatch.setattr(plugin.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(plugin.torch.cuda, 'device_count', lambda: 0)

    metadata = plugin._collect_environment_metadata()
    assert metadata['cuda_available'] is False
    assert metadata['cuda_device_count'] == 0
    assert 'torch_version' in metadata
