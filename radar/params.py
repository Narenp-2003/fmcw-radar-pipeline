from dataclasses import dataclass

C = 299_792_458.0  # m/s


@dataclass(frozen=True)
class RadarParams:
    fc: float = 77e9          # carrier, Hz
    bandwidth: float = 1e9    # Hz
    chirp_time: float = 40e-6 # s
    n_samples: int = 256      # per chirp (complex I/Q)
    n_chirps: int = 128       # per frame

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def slope(self) -> float:
        return self.bandwidth / self.chirp_time

    @property
    def fs(self) -> float:
        return self.n_samples / self.chirp_time

    @property
    def range_resolution(self) -> float:
        return C / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        return C * self.fs / (2 * self.slope)

    @property
    def max_velocity(self) -> float:
        return self.wavelength / (4 * self.chirp_time)

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2 * self.n_chirps * self.chirp_time)

    @property
    def frame_time(self) -> float:
        return self.n_chirps * self.chirp_time