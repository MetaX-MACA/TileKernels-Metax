import pytest


def test_get_num_sms_reads_env_override(monkeypatch):
    import tile_kernels.config as config

    monkeypatch.setattr(config, '_num_sms', 0)
    monkeypatch.setattr(config, 'get_device_num_sms', lambda: 128)
    monkeypatch.setenv('TILE_KERNELS_NUM_SMS', '64')

    assert config.get_num_sms() == 64
    monkeypatch.setenv('TILE_KERNELS_NUM_SMS', '32')
    assert config.get_num_sms() == 64


def test_get_num_sms_rejects_invalid_env(monkeypatch):
    import tile_kernels.config as config

    monkeypatch.setattr(config, '_num_sms', 0)
    monkeypatch.setattr(config, 'get_device_num_sms', lambda: 128)
    monkeypatch.setenv('TILE_KERNELS_NUM_SMS', '0')

    with pytest.raises(ValueError, match='between 1 and 128'):
        config.get_num_sms()
