from radar.params import C, RadarParams


def beat_frequency(p: RadarParams, range_m: float) -> float:
    """Beat frequency (Hz) for a target at range_m: 2*S*R/c."""
    return 2 * p.slope * range_m / C


def doppler_frequency(p: RadarParams, velocity_mps: float) -> float:
    """Doppler frequency (Hz) for radial velocity v: 2*v/lambda."""
    return 2 * velocity_mps / p.wavelength