import numpy as np


def _normalize_noise(noise: np.ndarray) -> np.ndarray:
    noise = noise - float(noise.mean())
    std = float(noise.std()) + 1e-12
    return noise / std


def correlated_noise_3d(shape, seed: int = 2026, fc_ratio: float = 0.18) -> np.ndarray:
    """
    Generate correlated 3D noise for seismic-like cubes (H, IL, XL).
    Correlation is introduced along time (frequency shaping) and laterally
    (neighbor averaging in inline/xline).
    """
    H, IL, XL = shape
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal(size=(H, IL, XL)).astype(np.float32)

    # Temporal shaping: low-pass filter in frequency domain (axis=0).
    freq = np.fft.rfftfreq(H, d=1.0)
    f_max = float(freq.max()) + 1e-12
    f0 = max(float(fc_ratio), 1e-3) * f_max
    filt = np.exp(-0.5 * (freq / f0) ** 2).astype(np.float32)[:, None, None]
    n_fft = np.fft.rfft(white, axis=0)
    n_time = np.fft.irfft(n_fft * filt, n=H, axis=0).astype(np.float32)

    # Lateral smoothing: simple neighborhood blend (periodic via roll).
    n_lat = (
        4.0 * n_time
        + np.roll(n_time, 1, axis=1)
        + np.roll(n_time, -1, axis=1)
        + np.roll(n_time, 1, axis=2)
        + np.roll(n_time, -1, axis=2)
    ) / 8.0

    return _normalize_noise(0.6 * n_time + 0.4 * n_lat).astype(np.float32)


def white_noise_3d(shape, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(size=shape).astype(np.float32)
    return _normalize_noise(noise).astype(np.float32)


def add_noise_by_std_ratio(
    seismic_3d: np.ndarray,
    noise_ratio: float,
    mode: str = "correlated",
    seed: int = 2026,
    fc_ratio: float = 0.18,
):
    """
    noise_ratio is relative to global seismic std:
      sigma_noise = noise_ratio * std(seismic)
    """
    seismic = np.asarray(seismic_3d, dtype=np.float32)
    if mode == "correlated":
        noise = correlated_noise_3d(seismic.shape, seed=seed, fc_ratio=fc_ratio)
    else:
        noise = white_noise_3d(seismic.shape, seed=seed)

    target_std = float(np.std(seismic)) * float(noise_ratio)
    noisy = seismic + noise * target_std
    return noisy.astype(np.float32), {
        "mode": mode,
        "noise_ratio": float(noise_ratio),
        "signal_std": float(np.std(seismic)),
        "target_noise_std": float(target_std),
        "real_noise_std": float(np.std(noisy - seismic)),
    }


def add_noise_by_snr_db(
    seismic_3d: np.ndarray,
    snr_db: float,
    mode: str = "correlated",
    seed: int = 2026,
    fc_ratio: float = 0.18,
):
    """
    Scale noise to achieve target SNR in dB:
      SNR = 10*log10(P_signal / P_noise)
    """
    seismic = np.asarray(seismic_3d, dtype=np.float32)
    if mode == "correlated":
        noise = correlated_noise_3d(seismic.shape, seed=seed, fc_ratio=fc_ratio)
    else:
        noise = white_noise_3d(seismic.shape, seed=seed)

    p_signal = float(np.mean(seismic**2)) + 1e-12
    p_noise_target = p_signal / (10.0 ** (float(snr_db) / 10.0))
    p_noise_now = float(np.mean(noise**2)) + 1e-12
    scale = float(np.sqrt(p_noise_target / p_noise_now))
    noisy = seismic + noise * scale

    p_noise_real = float(np.mean((noisy - seismic) ** 2)) + 1e-12
    snr_real = 10.0 * np.log10(p_signal / p_noise_real)
    return noisy.astype(np.float32), {
        "mode": mode,
        "snr_db_target": float(snr_db),
        "snr_db_real": float(snr_real),
        "signal_power": float(p_signal),
        "target_noise_power": float(p_noise_target),
        "real_noise_power": float(p_noise_real),
    }
