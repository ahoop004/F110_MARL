"""RGB capture keeps the public pixel layout without Pyglet's channel conversion."""
import importlib
from types import SimpleNamespace

import numpy as np


def test_rgb_capture_flips_rows_drops_alpha_and_owns_pixels(monkeypatch):
    module = importlib.import_module('env.f110ParallelEnv')
    pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    native = bytearray(pixels.tobytes())

    def get_data(format, pitch):
        assert (format, pitch) == ('RGBA', 12)
        return native

    buffer = SimpleNamespace(width=3, height=2,
                             get_image_data=lambda: SimpleNamespace(get_data=get_data))
    monkeypatch.setattr(module, '_ensure_pyglet', lambda: True)
    monkeypatch.setattr(module, 'pyg_img', SimpleNamespace(
        get_buffer_manager=lambda: SimpleNamespace(get_color_buffer=lambda: buffer)))
    monkeypatch.setattr(module, 'flush_render_state', lambda *args, **kwargs: None)
    env = module.F110ParallelEnv.__new__(module.F110ParallelEnv)
    env.render_mode = 'rgb_array'
    env._headless = True
    env._render_state = None
    env.renderer = SimpleNamespace(dispatch_events=lambda: None,
                                   on_draw=lambda: None, flip=lambda: None)
    frame = env.render()
    np.testing.assert_array_equal(frame, pixels[::-1, :, :3])
    assert frame.flags.c_contiguous and frame.flags.owndata
    native[:] = bytes(len(native))
    np.testing.assert_array_equal(frame, pixels[::-1, :, :3])
