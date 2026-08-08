import functools
import os
import torch

_num_sms = 0


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.multi_processor_count


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        env_num_sms = os.environ.get('TILE_KERNELS_NUM_SMS')
        if env_num_sms:
            try:
                value = int(env_num_sms)
            except ValueError as exc:
                raise ValueError('TILE_KERNELS_NUM_SMS must be an integer') from exc
            if not 0 < value <= get_device_num_sms():
                raise ValueError(
                    f'TILE_KERNELS_NUM_SMS must be between 1 and {get_device_num_sms()}, got {value}'
                )
            _num_sms = value
            return _num_sms
        _num_sms = get_device_num_sms()
        return _num_sms
    return _num_sms


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.shared_memory_per_multiprocessor
