"""Emergency message relay: store-and-forward messaging between phones when the phone network is down.

Phones walk around a disaster area. Two phones within radio range swap messages, so a message can hop from phone
to phone until one passes it to the rescue centre (a fixed phone in the middle). Compares three routing rules
(direct, epidemic, spray-and-wait) as phones go offline, and shows what a CRC checksum prevents when links corrupt data.

    python relay_sim.py        # runs both experiments, prints tables, saves results/*.png
"""
import zlib
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

AREA, RANGE, N_NODES, DT = 1000.0, 100.0, 40, 5  # metres, metres, phones (node 0 is the rescue centre), seconds per step
SIM_S, N_MSGS, CREATE_WINDOW_S, L_COPIES = 3600, 100, 1200, 8
SPEED = (1.0, 5.0)  # m/s, walking to cycling
CENTRE = 0


def make_frame(payload):
    return payload + zlib.crc32(payload).to_bytes(4, "big")


def frame_ok(frame):
    return zlib.crc32(frame[:-4]) == int.from_bytes(frame[-4:], "big")


def flip_bit(frame, rng):
    i = int(rng.integers(len(frame) * 8))
    b = bytearray(frame)
    b[i // 8] ^= 1 << (i % 8)
    return bytes(b)


def simulate(router, fail_frac=0.0, p_err=0.05, use_crc=True, seed=0, sim_s=SIM_S):
    """One run. router: 'direct' | 'epidemic' | 'spray'. fail_frac: share of phones that are dead for the whole run.

    p_err: chance that one transmission flips one random bit. Returns delivery, corruption, delay and cost figures.
    shortcut: unlimited phone storage and no deletion of delivered messages; add buffer limits if you study congestion.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, AREA, (N_NODES, 2))
    tgt = rng.uniform(0, AREA, (N_NODES, 2))
    spd = rng.uniform(*SPEED, N_NODES)  # random-waypoint mobility: walk to a random spot, then pick another

    alive = np.ones(N_NODES, dtype=bool)
    alive[rng.choice(np.arange(1, N_NODES), int(round(fail_frac * (N_NODES - 1))), replace=False)] = False
    sources = np.flatnonzero(alive[1:]) + 1
    payloads = [rng.bytes(32) for _ in range(N_MSGS)]
    times = np.sort(rng.uniform(0, CREATE_WINDOW_S, N_MSGS))
    senders = rng.choice(sources, N_MSGS) if sources.size else []

    store = [dict() for _ in range(N_NODES)]  # node -> {message id: [frame, copies left]}
    delivered = {}  # message id -> (time, payload arrived intact?)
    tx = 0
    t = 0

    def transfer(frame, dst, mid, copies):
        nonlocal tx
        tx += 1
        if rng.random() < p_err:
            frame = flip_bit(frame, rng)
        if use_crc and not frame_ok(frame):
            return False  # receiver drops it; the sender still has the message and tries again later
        if dst == CENTRE:
            delivered[mid] = (t, frame[:-4] == payloads[mid])
        else:
            store[dst][mid] = [frame, copies]
        return True

    def offer(x, y):
        for mid, held in store[x].items():
            frame, copies = held
            if y == CENTRE:
                if mid not in delivered:
                    transfer(frame, y, mid, 0)
            elif mid in store[y]:
                continue
            elif router == "epidemic":
                transfer(frame, y, mid, 1)
            elif router == "spray" and copies > 1:
                give = copies // 2
                if transfer(frame, y, mid, give):
                    held[1] -= give

    nxt = 0
    for step in range(int(sim_s / DT) if sources.size else 0):
        t = step * DT
        while nxt < N_MSGS and times[nxt] <= t:
            store[senders[nxt]][nxt] = [make_frame(payloads[nxt]), L_COPIES if router == "spray" else 1]
            nxt += 1

        vec = tgt - pos
        dist = np.linalg.norm(vec, axis=1)
        move = spd * DT
        arrived = dist <= move
        pos = np.where(arrived[:, None], tgt, pos + vec / np.maximum(dist, 1e-9)[:, None] * move[:, None])
        k = int(arrived.sum())
        tgt[arrived] = rng.uniform(0, AREA, (k, 2))
        spd[arrived] = rng.uniform(*SPEED, k)
        pos[CENTRE] = AREA / 2

        d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        for a, b in zip(*np.nonzero(np.triu((d < RANGE) & alive[:, None] & alive[None, :], 1))):
            if store[a] or store[b]:
                offer(a, b)
                offer(b, a)

    good = [tt - times[m] for m, (tt, ok) in delivered.items() if ok]
    return {
        "delivered": len(good) / N_MSGS,
        "corrupt": sum(not ok for _, ok in delivered.values()) / N_MSGS,
        "delay_min": float(np.mean(good)) / 60 if good else float("nan"),
        "tx_per_delivery": tx / len(good) if good else float("nan"),
    }


def _run(args):
    router, fail, p_err, crc, seed = args
    return simulate(router, fail, p_err, crc, seed)


def sweep(configs, seeds, pool):
    """configs: list of (router, fail_frac, p_err, use_crc). Returns {metric: array[config, seed]}."""
    jobs = [(*c, s) for c, s in product(configs, seeds)]
    res = list(pool.map(_run, jobs))
    keys = res[0].keys()
    return {k: np.array([r[k] for r in res]).reshape(len(configs), len(seeds)) for k in keys}


def plot(exp1, exp2, fails, perrs, routers, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"

    def style(axes):
        for ax in np.atleast_1d(axes):
            ax.set_facecolor(SURF)
            ax.grid(True, color="#e4e3df", lw=0.8)
            ax.tick_params(colors=MUTED)
            ax.xaxis.label.set_color(MUTED)
            ax.yaxis.label.set_color(MUTED)
            title = ax.get_title()
            ax.set_title("")
            ax.set_title(title, loc="left", color=INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    def line(ax, x, y, colour, marker, label):
        ax.plot(x, np.nanmean(y, axis=1), marker + "-", color=colour, lw=2, ms=8, mec=SURF, mew=2, label=label)
        ax.fill_between(x, np.nanmean(y, axis=1) - np.nanstd(y, axis=1), np.nanmean(y, axis=1) + np.nanstd(y, axis=1), color=colour, alpha=0.15, lw=0)

    names = {"direct": "direct delivery", "epidemic": "epidemic flooding", "spray": "spray and wait (8 copies)"}
    colours = {"direct": BLUE, "epidemic": ORANGE, "spray": AQUA}
    marks = {"direct": "o", "epidemic": "s", "spray": "^"}
    x = [100 * f for f in fails]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10), facecolor=SURF)
    for ax, key, title, ylab, scale in (
        (axes[0], "delivered", "Messages that reach the rescue centre", "Delivered (%)", 100),
        (axes[1], "delay_min", "Average delay of delivered messages", "Minutes", 1),
        (axes[2], "tx_per_delivery", "Cost: radio transmissions per delivered message", "Transmissions", 1),
    ):
        for i, r in enumerate(routers):
            line(ax, x, exp1[key][i * len(fails):(i + 1) * len(fails)] * scale, colours[r], marks[r], names[r])
        ax.set(xlabel="Phones switched off or out of battery (%)", ylabel=ylab, title=title)
        ax.legend(frameon=False, labelcolor=INK)
    style(axes)
    axes[0].set_ylim(0, 105)
    fig.text(0.01, 0.005, "Mean of 10 runs per point, band = one standard deviation. 40 phones, 1 km x 1 km, 100 m radio range, 1 hour.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_dir / "relay_routing.png", dpi=150)

    x = [100 * p for p in perrs]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), facecolor=SURF)
    for ax, key, title, ylab in (
        (axes[0], "delivered", "Messages delivered intact", "Delivered intact (%)"),
        (axes[1], "corrupt", "Messages delivered with damaged contents", "Delivered damaged (%)"),
    ):
        for j, (label, colour, marker) in enumerate((("with CRC check", BLUE, "o"), ("no check", ORANGE, "s"))):
            line(ax, x, exp2[key][j * len(perrs):(j + 1) * len(perrs)] * 100, colour, marker, label)
        ax.set(xlabel="Chance that one transmission flips a bit (%)", ylabel=ylab, title=title)
        ax.legend(frameon=False, labelcolor=INK)
    style(axes)
    axes[1].set_ylim(bottom=-2)
    fig.text(0.01, 0.005, "Epidemic routing, mean of 10 runs per point, band = one standard deviation.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_dir / "relay_crc.png", dpi=150)


def main():
    routers, fails, perrs, seeds = ["direct", "epidemic", "spray"], [0, 0.2, 0.4, 0.6, 0.8], [0, 0.05, 0.1, 0.2, 0.3], range(10)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as pool:
        exp1 = sweep([(r, f, 0.05, True) for r in routers for f in fails], seeds, pool)
        exp2 = sweep([("epidemic", 0, p, crc) for crc in (True, False) for p in perrs], seeds, pool)

    print("Experiment 1: routing rules as phones go offline (5% chance a transmission is corrupted, CRC on)")
    print(f"{'router':<10}{'offline':>8}{'delivered':>11}{'delay min':>11}{'tx/msg':>9}")
    for i, (r, f) in enumerate((r, f) for r in routers for f in fails):
        print(f"{r:<10}{f:>8.0%}{exp1['delivered'][i].mean():>11.0%}{np.nanmean(exp1['delay_min'][i]):>11.1f}{np.nanmean(exp1['tx_per_delivery'][i]):>9.1f}")
    print("\nExperiment 2: epidemic routing, CRC on vs off")
    print(f"{'CRC':<6}{'bit-flip chance':>16}{'intact':>9}{'damaged':>9}")
    for i, (crc, p) in enumerate((c, p) for c in ("on", "off") for p in perrs):
        print(f"{crc:<6}{p:>16.0%}{exp2['delivered'][i].mean():>9.0%}{exp2['corrupt'][i].mean():>9.0%}")
    plot(exp1, exp2, fails, perrs, routers, out)
    print("saved", out)


if __name__ == "__main__":
    main()
