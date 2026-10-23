"""Run with `python -m pytest -q`."""
import numpy as np

from relay_sim import frame_ok, make_frame, simulate

SHORT = 1800  # half-hour runs keep the tests quick


def test_crc_detects_every_single_bit_flip():
    frame = make_frame(np.random.default_rng(0).bytes(32))
    assert frame_ok(frame)
    for i in range(len(frame) * 8):
        damaged = bytearray(frame)
        damaged[i // 8] ^= 1 << (i % 8)
        assert not frame_ok(bytes(damaged)), i


def test_crc_lets_no_damaged_message_through():
    r = simulate("epidemic", p_err=0.3, use_crc=True, seed=1, sim_s=SHORT)
    assert r["corrupt"] == 0 and r["delivered"] > 0


def test_without_crc_damaged_messages_get_through():
    assert simulate("epidemic", p_err=0.3, use_crc=False, seed=1, sim_s=SHORT)["corrupt"] > 0


def test_same_seed_gives_same_result():
    a = simulate("spray", fail_frac=0.2, seed=5, sim_s=SHORT)
    b = simulate("spray", fail_frac=0.2, seed=5, sim_s=SHORT)
    assert a == b or all(np.isclose(a[k], b[k], equal_nan=True) for k in a)


def test_nothing_is_delivered_when_every_phone_is_dead():
    assert simulate("epidemic", fail_frac=1.0, sim_s=SHORT)["delivered"] == 0


def test_flooding_delivers_at_least_as_much_as_direct_on_average():
    seeds = range(4)
    flood = np.mean([simulate("epidemic", seed=s, sim_s=SHORT)["delivered"] for s in seeds])
    direct = np.mean([simulate("direct", seed=s, sim_s=SHORT)["delivered"] for s in seeds])
    assert flood >= direct, (flood, direct)
