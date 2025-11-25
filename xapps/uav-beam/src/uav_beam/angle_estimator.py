"""
Angle of Arrival (AoA) / Angle of Departure (AoD) Estimator

High-fidelity angle estimation for mmWave beam alignment using:
1. 2D MUSIC (Multiple Signal Classification) with spatial smoothing
2. 2D ESPRIT for UPA (Uniform Planar Array)
3. Unitary ESPRIT for improved precision
4. Root-MUSIC for fast 1D estimation
5. Beamspace methods for real-time operation

Optimizations applied:
- NumPy vectorized operations
- Pre-computed steering vectors
- scipy.linalg optimized matrix operations
- LRU cache for repeated computations
- Optional Numba JIT compilation for hot paths

References:
- R. Schmidt, "Multiple emitter location and signal parameter estimation"
- R. Roy and T. Kailath, "ESPRIT - Estimation of Signal Parameters via
  Rotational Invariance Techniques"
- M. Haardt and J. A. Nossek, "Unitary ESPRIT: How to obtain increased
  estimation accuracy with a reduced computational burden"
"""

import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)


class EstimationMethod(Enum):
    """Angle estimation method"""
    MUSIC_2D = "music_2d"                 # Full 2D MUSIC with spatial smoothing
    MUSIC_1D = "music_1d"                 # 1D MUSIC (azimuth only)
    ROOT_MUSIC = "root_music"             # Polynomial root finding
    ESPRIT_2D = "esprit_2d"               # 2D ESPRIT for UPA
    ESPRIT_1D = "esprit_1d"               # Standard 1D ESPRIT
    UNITARY_ESPRIT = "unitary_esprit"     # Unitary ESPRIT (higher precision)
    BEAMSPACE = "beamspace"               # Beamspace method (fast)
    # Legacy aliases
    MUSIC = "music"
    ESPRIT = "esprit"
    ML = "ml"


@dataclass
class AngleEstimatorConfig:
    """Angle estimator configuration"""
    # Antenna array parameters
    num_elements_h: int = 8               # Horizontal antenna elements (M)
    num_elements_v: int = 8               # Vertical antenna elements (N)
    element_spacing: float = 0.5          # Element spacing in wavelengths

    # Signal parameters
    num_snapshots: int = 64               # Number of signal snapshots (K)
    num_sources: int = 1                  # Expected number of signal sources
    subspace_dimension: int = 4           # Signal subspace dimension

    # Subspace estimation
    subspace_method: str = "mdl"          # "mdl", "aic", or "fixed"

    # Angular search grid
    azimuth_range: Tuple[float, float] = (-np.pi/2, np.pi/2)
    elevation_range: Tuple[float, float] = (-np.pi/4, np.pi/4)
    angular_resolution: float = 0.02      # radians (~1 degree)

    # Spatial smoothing (for coherent sources)
    enable_spatial_smoothing: bool = True
    smoothing_subarray_h: int = 6         # Horizontal subarray size
    smoothing_subarray_v: int = 6         # Vertical subarray size

    # Forward-Backward averaging
    enable_fb_averaging: bool = True

    # Noise parameters
    noise_power_dbm: float = -100.0
    snr_threshold_db: float = 5.0

    # Optimization parameters
    use_svd_for_music: bool = True
    cache_steering_vectors: bool = True


@dataclass
class AngleEstimate:
    """Angle estimation result"""
    timestamp_ms: float
    azimuth_rad: float                    # Horizontal angle (theta)
    elevation_rad: float                  # Vertical angle (phi)
    confidence: float                     # Estimation confidence [0, 1]
    method: str                           # Method used

    # Optional detailed results
    spectrum: Optional[np.ndarray] = None
    all_estimates: List[Tuple[float, float]] = field(default_factory=list)
    snr_db: float = 0.0
    num_sources_detected: int = 1


# Numba-optimized steering vector computation
@jit(nopython=True, cache=True, fastmath=True)
def _compute_steering_vector_numba(
    azimuth: float,
    elevation: float,
    num_h: int,
    num_v: int,
    d: float
) -> np.ndarray:
    """Compute steering vector using Numba JIT"""
    num_elements = num_h * num_v
    sv = np.zeros(num_elements, dtype=np.complex128)

    u = np.sin(azimuth) * np.cos(elevation)
    v = np.sin(elevation)

    elem_idx = 0
    for m in range(num_h):
        for n in range(num_v):
            phase = 2.0 * np.pi * d * (m * u + n * v)
            sv[elem_idx] = np.exp(1j * phase)
            elem_idx += 1

    norm = np.sqrt(np.sum(np.abs(sv)**2))
    return sv / norm


# Numba-optimized MUSIC spectrum computation
@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_music_spectrum_numba(
    noise_proj: np.ndarray,
    array_response: np.ndarray,
    num_az: int,
    num_el: int
) -> np.ndarray:
    """Compute MUSIC spectrum using Numba parallel JIT"""
    spectrum = np.zeros((num_az, num_el))

    for az_idx in prange(num_az):
        for el_idx in range(num_el):
            sv = array_response[az_idx, el_idx, :]
            denom = np.abs(np.vdot(sv, noise_proj @ sv))
            spectrum[az_idx, el_idx] = 1.0 / (denom + 1e-10)

    return spectrum


class AngleEstimator:
    """
    AoA/AoD Angle Estimator for mmWave Systems

    Implements high-resolution angle estimation algorithms:
    - 2D MUSIC with proper noise subspace computation
    - 2D ESPRIT for UPA with azimuth and elevation estimation
    - Unitary ESPRIT for improved accuracy
    - Spatial smoothing for coherent source handling
    - Forward-backward averaging
    - Automatic source number detection (MDL/AIC)

    Optimizations:
    - Pre-computed array response matrix
    - Vectorized NumPy operations
    - scipy.linalg for optimized eigendecomposition
    - Optional Numba JIT for hot paths
    - LRU cache for steering vectors
    """

    def __init__(self, config: Optional[AngleEstimatorConfig] = None):
        self.config = config or AngleEstimatorConfig()

        # Angular search grids
        self.azimuth_grid = np.arange(
            self.config.azimuth_range[0],
            self.config.azimuth_range[1],
            self.config.angular_resolution
        )
        self.elevation_grid = np.arange(
            self.config.elevation_range[0],
            self.config.elevation_range[1],
            self.config.angular_resolution
        )

        # Pre-compute array manifold (vectorized)
        self.array_response = self._compute_array_response_vectorized()

        # Pre-compute DFT matrices for beamspace
        self._dft_h = np.fft.fft(np.eye(self.config.num_elements_h)) / np.sqrt(self.config.num_elements_h)
        self._dft_v = np.fft.fft(np.eye(self.config.num_elements_v)) / np.sqrt(self.config.num_elements_v)
        self._dft_codebook = np.kron(self._dft_h, self._dft_v)

        # Statistics
        self.stats = {
            "estimates": 0,
            "avg_confidence": 0.0,
            "method_usage": {},
        }

        logger.info(
            f"AngleEstimator initialized: {self.config.num_elements_h}x"
            f"{self.config.num_elements_v} UPA (Numba: {NUMBA_AVAILABLE})"
        )

    @property
    def num_elements(self) -> int:
        return self.config.num_elements_h * self.config.num_elements_v

    def _compute_array_response_vectorized(self) -> np.ndarray:
        """Pre-compute array response using vectorized NumPy operations"""
        num_az = len(self.azimuth_grid)
        num_el = len(self.elevation_grid)
        num_h = self.config.num_elements_h
        num_v = self.config.num_elements_v
        d = self.config.element_spacing

        az_mesh, el_mesh = np.meshgrid(self.azimuth_grid, self.elevation_grid, indexing='ij')

        sin_az = np.sin(az_mesh)
        cos_el = np.cos(el_mesh)
        sin_el = np.sin(el_mesh)

        u = sin_az * cos_el
        v = sin_el

        m_indices = np.arange(num_h)
        n_indices = np.arange(num_v)

        m_grid = m_indices.reshape(1, 1, -1, 1)
        n_grid = n_indices.reshape(1, 1, 1, -1)

        phase = 2 * np.pi * d * (
            m_grid * u[:, :, np.newaxis, np.newaxis] +
            n_grid * v[:, :, np.newaxis, np.newaxis]
        )

        array_response = np.exp(1j * phase).reshape(num_az, num_el, -1)

        return array_response

    @lru_cache(maxsize=2048)
    def steering_vector_cached(self, azimuth: float, elevation: float) -> tuple:
        """Cached steering vector computation"""
        sv = self.steering_vector(azimuth, elevation)
        return tuple(sv)

    def steering_vector(self, azimuth: float, elevation: float) -> np.ndarray:
        """Compute steering vector for UPA"""
        if NUMBA_AVAILABLE:
            return _compute_steering_vector_numba(
                azimuth, elevation,
                self.config.num_elements_h,
                self.config.num_elements_v,
                self.config.element_spacing
            )

        d = self.config.element_spacing
        num_h = self.config.num_elements_h
        num_v = self.config.num_elements_v

        u = np.sin(azimuth) * np.cos(elevation)
        v = np.sin(elevation)

        m = np.arange(num_h)
        n = np.arange(num_v)

        a_h = np.exp(1j * 2 * np.pi * d * m * u)
        a_v = np.exp(1j * 2 * np.pi * d * n * v)

        sv = np.kron(a_h, a_v)
        return sv / np.linalg.norm(sv)

    # =========================================================================
    # Main Estimation Interface
    # =========================================================================

    def estimate(
        self,
        received_signal: np.ndarray,
        timestamp_ms: float,
        method: EstimationMethod = EstimationMethod.MUSIC_2D
    ) -> AngleEstimate:
        """
        Estimate AoA/AoD from received signal

        Args:
            received_signal: Received signal matrix (num_elements, num_snapshots)
            timestamp_ms: Timestamp of measurement
            method: Estimation method to use

        Returns:
            AngleEstimate with estimated angles
        """
        # Adjust dimensions if needed
        if received_signal.shape[0] != self.num_elements:
            received_signal = self._adjust_signal_dimensions(received_signal)

        # Method dispatch
        if method in [EstimationMethod.MUSIC_2D, EstimationMethod.MUSIC]:
            result = self._estimate_music_2d(received_signal, timestamp_ms)
        elif method == EstimationMethod.MUSIC_1D:
            result = self._estimate_music_1d(received_signal, timestamp_ms)
        elif method == EstimationMethod.ROOT_MUSIC:
            result = self._estimate_root_music(received_signal, timestamp_ms)
        elif method in [EstimationMethod.ESPRIT_2D, EstimationMethod.ESPRIT]:
            result = self._estimate_esprit_2d(received_signal, timestamp_ms)
        elif method == EstimationMethod.ESPRIT_1D:
            result = self._estimate_esprit_1d(received_signal, timestamp_ms)
        elif method == EstimationMethod.UNITARY_ESPRIT:
            result = self._estimate_unitary_esprit(received_signal, timestamp_ms)
        elif method == EstimationMethod.BEAMSPACE:
            result = self._estimate_beamspace(received_signal, timestamp_ms)
        else:
            result = self._estimate_music_2d(received_signal, timestamp_ms)

        self._update_stats(result)
        return result

    def _adjust_signal_dimensions(self, received_signal: np.ndarray) -> np.ndarray:
        """Adjust signal dimensions to match array size"""
        if received_signal.shape[0] > self.num_elements:
            return received_signal[:self.num_elements, :]
        else:
            padded = np.zeros((self.num_elements, received_signal.shape[1]), dtype=complex)
            padded[:received_signal.shape[0], :] = received_signal
            return padded

    # =========================================================================
    # Covariance Matrix Computation
    # =========================================================================

    def _compute_covariance(self, received_signal: np.ndarray, apply_fb: bool = True) -> np.ndarray:
        """Compute sample covariance with forward-backward averaging"""
        K = received_signal.shape[1]
        R = received_signal @ received_signal.conj().T / K

        if apply_fb and self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(R.shape[0]))
            R = (R + J @ R.conj() @ J) / 2

        return R

    def _compute_smoothed_covariance(self, received_signal: np.ndarray) -> np.ndarray:
        """Compute spatially smoothed covariance for coherent sources"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v
        L_h = self.config.smoothing_subarray_h
        L_v = self.config.smoothing_subarray_v

        X = received_signal.reshape(M, N, -1)
        K = X.shape[2]

        num_subarrays_h = M - L_h + 1
        num_subarrays_v = N - L_v + 1
        subarray_size = L_h * L_v

        R_smooth = np.zeros((subarray_size, subarray_size), dtype=complex)

        for i_h in range(num_subarrays_h):
            for i_v in range(num_subarrays_v):
                subarray = X[i_h:i_h+L_h, i_v:i_v+L_v, :]
                subarray_flat = subarray.reshape(subarray_size, K)
                R_smooth += subarray_flat @ subarray_flat.conj().T / K

        R_smooth /= (num_subarrays_h * num_subarrays_v)

        if self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(subarray_size))
            R_smooth = (R_smooth + J @ R_smooth.conj() @ J) / 2

        return R_smooth

    # =========================================================================
    # Source Number Estimation (MDL/AIC)
    # =========================================================================

    def _estimate_num_sources(self, eigenvalues: np.ndarray, num_snapshots: int) -> int:
        """Estimate number of sources using MDL or AIC"""
        if self.config.subspace_method == "fixed":
            return min(self.config.subspace_dimension, len(eigenvalues) - 1)

        M = len(eigenvalues)
        N = num_snapshots
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        criteria = []
        for k in range(M - 1):
            noise_eigs = eigenvalues[k+1:]
            if len(noise_eigs) == 0:
                continue

            log_geo_mean = np.mean(np.log(noise_eigs))
            geo_mean = np.exp(log_geo_mean)
            arith_mean = np.mean(noise_eigs)

            if geo_mean > 0 and arith_mean > 0:
                ratio = geo_mean / arith_mean
                log_ratio = (M - k - 1) * np.log(ratio + 1e-10)
            else:
                log_ratio = 0

            if self.config.subspace_method == "mdl":
                penalty = 0.5 * (k + 1) * (2 * M - k - 1) * np.log(N)
                criterion = -N * (M - k - 1) * log_ratio + penalty
            else:
                penalty = (k + 1) * (2 * M - k - 1)
                criterion = -2 * N * (M - k - 1) * log_ratio + 2 * penalty

            criteria.append(criterion)

        if criteria:
            num_sources = np.argmin(criteria) + 1
        else:
            num_sources = 1

        return min(num_sources, self.config.num_sources)

    # =========================================================================
    # 2D MUSIC Algorithm
    # =========================================================================

    def _estimate_music_2d(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """2D MUSIC with spatial smoothing and proper noise subspace"""
        if self.config.enable_spatial_smoothing:
            R = self._compute_smoothed_covariance(received_signal)
        else:
            R = self._compute_covariance(received_signal)

        eigenvalues, eigenvectors = linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        num_sources = self._estimate_num_sources(eigenvalues, received_signal.shape[1])
        En = eigenvectors[:, num_sources:]
        noise_proj = En @ En.conj().T

        if self.config.enable_spatial_smoothing:
            spectrum = self._compute_music_spectrum_smoothed(noise_proj)
        else:
            spectrum = self._compute_music_spectrum_vectorized(noise_proj)

        spectrum = spectrum / (spectrum.max() + 1e-10)
        peaks = self._find_spectrum_peaks(spectrum)

        if peaks:
            best_peak = peaks[0]
            azimuth_est = self.azimuth_grid[best_peak[0]]
            elevation_est = self.elevation_grid[best_peak[1]]
            confidence = self._compute_confidence_from_spectrum(spectrum, best_peak)
            all_estimates = [(self.azimuth_grid[p[0]], self.elevation_grid[p[1]]) for p in peaks]
        else:
            peak_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
            azimuth_est = self.azimuth_grid[peak_idx[0]]
            elevation_est = self.elevation_grid[peak_idx[1]]
            confidence = 0.5
            all_estimates = [(azimuth_est, elevation_est)]

        if num_sources > 0 and len(eigenvalues) > num_sources:
            signal_power = np.mean(eigenvalues[:num_sources])
            noise_power = np.mean(eigenvalues[num_sources:])
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        else:
            snr_db = 0.0

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=confidence,
            method="MUSIC-2D",
            spectrum=spectrum,
            all_estimates=all_estimates,
            snr_db=snr_db,
            num_sources_detected=num_sources
        )

    def _compute_music_spectrum_smoothed(self, noise_proj: np.ndarray) -> np.ndarray:
        """Compute MUSIC spectrum for smoothed covariance"""
        L_h = self.config.smoothing_subarray_h
        L_v = self.config.smoothing_subarray_v
        d = self.config.element_spacing

        num_az = len(self.azimuth_grid)
        num_el = len(self.elevation_grid)
        spectrum = np.zeros((num_az, num_el))

        for i_az, az in enumerate(self.azimuth_grid):
            for i_el, el in enumerate(self.elevation_grid):
                u = np.sin(az) * np.cos(el)
                v = np.sin(el)

                m = np.arange(L_h)
                n = np.arange(L_v)

                a_h = np.exp(1j * 2 * np.pi * d * m * u)
                a_v = np.exp(1j * 2 * np.pi * d * n * v)

                sv = np.kron(a_h, a_v)
                sv = sv / np.linalg.norm(sv)

                denom = np.abs(sv.conj() @ noise_proj @ sv)
                spectrum[i_az, i_el] = 1.0 / (denom + 1e-10)

        return spectrum

    def _compute_music_spectrum_vectorized(self, noise_proj: np.ndarray) -> np.ndarray:
        """Vectorized MUSIC spectrum computation"""
        sv_flat = self.array_response.reshape(-1, self.num_elements)
        projected = np.einsum('ij,jk,ik->i', sv_flat.conj(), noise_proj, sv_flat)
        spectrum = 1.0 / (np.abs(projected) + 1e-10)
        return spectrum.reshape(len(self.azimuth_grid), len(self.elevation_grid))

    # =========================================================================
    # 1D MUSIC and Root-MUSIC
    # =========================================================================

    def _estimate_music_1d(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """1D MUSIC for azimuth estimation"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v
        d = self.config.element_spacing

        X = received_signal.reshape(M, N, -1)
        X_h = X.mean(axis=1)

        R = X_h @ X_h.conj().T / X_h.shape[1]
        if self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(M))
            R = (R + J @ R.conj() @ J) / 2

        eigenvalues, eigenvectors = linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        num_sources = min(self.config.num_sources, M - 1)
        En = eigenvectors[:, num_sources:]
        noise_proj = En @ En.conj().T

        spectrum_1d = np.zeros(len(self.azimuth_grid))
        m = np.arange(M)

        for i_az, az in enumerate(self.azimuth_grid):
            sv = np.exp(1j * 2 * np.pi * d * m * np.sin(az))
            sv = sv / np.linalg.norm(sv)
            denom = np.abs(sv.conj() @ noise_proj @ sv)
            spectrum_1d[i_az] = 1.0 / (denom + 1e-10)

        peak_idx = np.argmax(spectrum_1d)
        azimuth_est = self.azimuth_grid[peak_idx]
        elevation_est = self._estimate_elevation_from_vertical(received_signal)
        confidence = self._compute_confidence_1d(spectrum_1d, peak_idx)

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=confidence,
            method="MUSIC-1D",
            spectrum=spectrum_1d.reshape(-1, 1),
            num_sources_detected=num_sources
        )

    def _estimate_root_music(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """Root-MUSIC for fast 1D estimation"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v

        X = received_signal.reshape(M, N, -1)
        X_h = X.mean(axis=1)

        R = X_h @ X_h.conj().T / X_h.shape[1]
        if self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(M))
            R = (R + J @ R.conj() @ J) / 2

        eigenvalues, eigenvectors = linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        num_sources = min(self.config.num_sources, M - 2)
        En = eigenvectors[:, num_sources:]
        C = En @ En.conj().T

        poly_order = 2 * (M - 1)
        coeffs = np.zeros(poly_order + 1, dtype=complex)
        for i in range(M):
            for j in range(M):
                k = (M - 1) + (i - j)
                coeffs[k] += C[i, j]

        roots = np.roots(coeffs)
        valid_roots = [r for r in roots if np.abs(r) <= 1.1]

        if not valid_roots:
            return self._estimate_music_1d(received_signal, timestamp_ms)

        valid_roots.sort(key=lambda r: abs(abs(r) - 1.0))
        best_root = valid_roots[0]

        if np.abs(best_root) > 0:
            best_root = best_root / np.abs(best_root)

        angle_arg = np.angle(best_root) / (2 * np.pi * self.config.element_spacing)
        angle_arg = np.clip(angle_arg, -1, 1)
        azimuth_est = np.arcsin(angle_arg)

        elevation_est = self._estimate_elevation_from_vertical(received_signal)
        root_quality = 1.0 - abs(abs(valid_roots[0]) - 1.0)
        confidence = min(0.95, root_quality)

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=confidence,
            method="Root-MUSIC",
            num_sources_detected=num_sources
        )

    def _estimate_elevation_from_vertical(self, received_signal: np.ndarray) -> float:
        """Estimate elevation using vertical array"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v
        d = self.config.element_spacing

        X = received_signal.reshape(M, N, -1)
        X_v = X.mean(axis=0)
        R = X_v @ X_v.conj().T / X_v.shape[1]

        best_power = -np.inf
        best_el = 0.0
        n = np.arange(N)

        for el in self.elevation_grid:
            sv = np.exp(1j * 2 * np.pi * d * n * np.sin(el))
            sv = sv / np.linalg.norm(sv)
            power = np.real(sv.conj() @ R @ sv)
            if power > best_power:
                best_power = power
                best_el = el

        return best_el

    # =========================================================================
    # 2D ESPRIT Algorithm
    # =========================================================================

    def _estimate_esprit_2d(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """2D ESPRIT for joint azimuth-elevation estimation"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v

        X = received_signal.reshape(M, N, -1)
        K = X.shape[2]

        # Azimuth (horizontal shift invariance)
        X_h = X.mean(axis=1)
        R_h = X_h @ X_h.conj().T / K

        if self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(M))
            R_h = (R_h + J @ R_h.conj() @ J) / 2

        eigenvalues_h, eigenvectors_h = linalg.eigh(R_h)
        idx = np.argsort(eigenvalues_h)[::-1]
        eigenvalues_h = eigenvalues_h[idx]
        eigenvectors_h = eigenvectors_h[:, idx]

        num_sources = min(self.config.num_sources, M - 2)
        Es_h = eigenvectors_h[:, :num_sources]

        J1_h = np.eye(M - 1, M)
        J2_h = np.eye(M - 1, M, 1)

        E1_h = J1_h @ Es_h
        E2_h = J2_h @ Es_h

        Phi_h, _, _, _ = linalg.lstsq(E1_h, E2_h)
        eig_h = linalg.eigvals(Phi_h)

        main_eig_h = eig_h[np.argmax(np.abs(eig_h))]
        phase_h = np.angle(main_eig_h)
        sin_az = phase_h / (2 * np.pi * self.config.element_spacing)
        sin_az = np.clip(sin_az, -1, 1)
        azimuth_est = np.arcsin(sin_az)

        # Elevation (vertical shift invariance)
        X_v = X.mean(axis=0)
        R_v = X_v @ X_v.conj().T / K

        if self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(N))
            R_v = (R_v + J @ R_v.conj() @ J) / 2

        eigenvalues_v, eigenvectors_v = linalg.eigh(R_v)
        idx = np.argsort(eigenvalues_v)[::-1]
        eigenvectors_v = eigenvectors_v[:, idx]

        Es_v = eigenvectors_v[:, :min(num_sources, N-2)]

        J1_v = np.eye(N - 1, N)
        J2_v = np.eye(N - 1, N, 1)

        E1_v = J1_v @ Es_v
        E2_v = J2_v @ Es_v

        Phi_v, _, _, _ = linalg.lstsq(E1_v, E2_v)
        eig_v = linalg.eigvals(Phi_v)

        main_eig_v = eig_v[np.argmax(np.abs(eig_v))]
        phase_v = np.angle(main_eig_v)
        sin_el = phase_v / (2 * np.pi * self.config.element_spacing)
        sin_el = np.clip(sin_el, -1, 1)
        elevation_est = np.arcsin(sin_el)

        if len(eigenvalues_h) > num_sources:
            signal_power = np.mean(eigenvalues_h[:num_sources])
            noise_power = np.mean(eigenvalues_h[num_sources:])
            snr = signal_power / (noise_power + 1e-10)
            confidence = min(0.95, 1 - 1/(snr + 1))
        else:
            confidence = 0.7

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=confidence,
            method="ESPRIT-2D",
            num_sources_detected=num_sources
        )

    def _estimate_esprit_1d(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """Standard 1D ESPRIT"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v

        X = received_signal.reshape(M, N, -1)
        X_h = X.mean(axis=1)

        R = X_h @ X_h.conj().T / X_h.shape[1]

        if self.config.enable_fb_averaging:
            J = np.fliplr(np.eye(M))
            R = (R + J @ R.conj() @ J) / 2

        eigenvalues, eigenvectors = linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        num_sources = min(self.config.num_sources, M - 2)
        Es = eigenvectors[:, :num_sources]

        J1 = np.eye(M - 1, M)
        J2 = np.eye(M - 1, M, 1)

        E1 = J1 @ Es
        E2 = J2 @ Es

        Phi, _, _, _ = linalg.lstsq(E1, E2)
        eig_vals = linalg.eigvals(Phi)

        main_eig = eig_vals[np.argmax(np.abs(eig_vals))]
        phase = np.angle(main_eig)
        sin_az = phase / (2 * np.pi * self.config.element_spacing)
        sin_az = np.clip(sin_az, -1, 1)
        azimuth_est = np.arcsin(sin_az)

        elevation_est = self._estimate_elevation_from_vertical(received_signal)

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=0.8,
            method="ESPRIT-1D",
            num_sources_detected=num_sources
        )

    # =========================================================================
    # Unitary ESPRIT
    # =========================================================================

    def _estimate_unitary_esprit(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """Unitary ESPRIT for improved accuracy"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v

        X = received_signal.reshape(M, N, -1)
        X_h = X.mean(axis=1)
        K = X_h.shape[1]

        Q_M = self._construct_unitary_matrix(M)

        X_real = Q_M.conj().T @ X_h
        R_real = X_real @ X_real.conj().T / K
        R_real = (R_real + R_real.conj().T) / 2

        eigenvalues, eigenvectors = linalg.eigh(R_real)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        num_sources = min(self.config.num_sources, M - 2)
        Es = eigenvectors[:, :num_sources]

        K1 = np.eye(M - 1, M)
        K2 = np.eye(M - 1, M, 1)

        E1 = K1 @ Es
        E2 = K2 @ Es

        Phi, _, _, _ = linalg.lstsq(E1, E2)
        eig_vals = linalg.eigvals(Phi)

        angles = []
        for eig in eig_vals:
            if np.abs(np.imag(eig)) < 1e-6:
                angle = 2 * np.arctan(np.real(eig))
                angles.append(angle)

        if angles:
            main_angle = angles[0]
            sin_az = main_angle / (2 * np.pi * self.config.element_spacing)
            sin_az = np.clip(sin_az, -1, 1)
            azimuth_est = np.arcsin(sin_az)
        else:
            return self._estimate_esprit_1d(received_signal, timestamp_ms)

        elevation_est = self._estimate_elevation_from_vertical(received_signal)

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=0.85,
            method="Unitary-ESPRIT",
            num_sources_detected=num_sources
        )

    def _construct_unitary_matrix(self, M: int) -> np.ndarray:
        """Construct unitary transformation matrix"""
        if M % 2 == 0:
            M_half = M // 2
            I = np.eye(M_half)
            J = np.fliplr(I)

            Q = np.zeros((M, M), dtype=complex)
            Q[:M_half, :M_half] = I
            Q[:M_half, M_half:] = 1j * I
            Q[M_half:, :M_half] = J
            Q[M_half:, M_half:] = -1j * J
        else:
            M_half = (M - 1) // 2
            I = np.eye(M_half)
            J = np.fliplr(I)

            Q = np.zeros((M, M), dtype=complex)
            Q[:M_half, :M_half] = I
            Q[:M_half, M_half+1:] = 1j * I
            Q[M_half, M_half] = np.sqrt(2)
            Q[M_half+1:, :M_half] = J
            Q[M_half+1:, M_half+1:] = -1j * J

        return Q / np.sqrt(2)

    # =========================================================================
    # Beamspace Method
    # =========================================================================

    def _estimate_beamspace(self, received_signal: np.ndarray, timestamp_ms: float) -> AngleEstimate:
        """Fast beamspace-based estimation using DFT"""
        M = self.config.num_elements_h
        N = self.config.num_elements_v

        beam_output = self._dft_codebook @ received_signal
        beam_power = np.mean(np.abs(beam_output) ** 2, axis=1)
        beam_power_2d = beam_power.reshape(M, N)

        max_idx = np.unravel_index(np.argmax(beam_power_2d), beam_power_2d.shape)
        h_idx, v_idx = max_idx

        spatial_freq_h = (h_idx - M / 2) / M
        sin_az = spatial_freq_h / self.config.element_spacing
        sin_az = np.clip(sin_az, -1, 1)
        azimuth_est = np.arcsin(sin_az)

        spatial_freq_v = (v_idx - N / 2) / N
        sin_el = spatial_freq_v / self.config.element_spacing
        sin_el = np.clip(sin_el, -1, 1)
        elevation_est = np.arcsin(sin_el)

        sorted_power = np.partition(beam_power, -2)[-2:]
        if sorted_power[1] > 0:
            confidence = min(0.9, (sorted_power[1] - sorted_power[0]) / sorted_power[1])
        else:
            confidence = 0.3

        return AngleEstimate(
            timestamp_ms=timestamp_ms,
            azimuth_rad=float(azimuth_est),
            elevation_rad=float(elevation_est),
            confidence=confidence,
            method="Beamspace",
            spectrum=beam_power_2d
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_spectrum_peaks(self, spectrum: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find peaks in 2D spectrum"""
        peaks = []
        for i in range(1, spectrum.shape[0] - 1):
            for j in range(1, spectrum.shape[1] - 1):
                val = spectrum[i, j]
                neighbors = [
                    spectrum[i-1, j], spectrum[i+1, j],
                    spectrum[i, j-1], spectrum[i, j+1],
                    spectrum[i-1, j-1], spectrum[i-1, j+1],
                    spectrum[i+1, j-1], spectrum[i+1, j+1]
                ]
                if val > max(neighbors) and val > threshold:
                    peaks.append((i, j, val))

        peaks.sort(key=lambda x: x[2], reverse=True)
        return [(p[0], p[1]) for p in peaks[:self.config.num_sources]]

    def _compute_confidence_from_spectrum(self, spectrum: np.ndarray, peak_idx: Tuple[int, int]) -> float:
        """Compute confidence from spectrum peak prominence"""
        peak_val = spectrum[peak_idx]

        mask = np.ones_like(spectrum, dtype=bool)
        r = 3
        i_min = max(0, peak_idx[0] - r)
        i_max = min(spectrum.shape[0], peak_idx[0] + r + 1)
        j_min = max(0, peak_idx[1] - r)
        j_max = min(spectrum.shape[1], peak_idx[1] + r + 1)
        mask[i_min:i_max, j_min:j_max] = False

        if mask.sum() > 0:
            background = spectrum[mask].mean()
        else:
            background = 0.1

        if peak_val > 0:
            prominence = (peak_val - background) / peak_val
            confidence = min(0.95, max(0.3, prominence))
        else:
            confidence = 0.3

        return confidence

    def _compute_confidence_1d(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Compute confidence from 1D spectrum"""
        peak_val = spectrum[peak_idx]

        mask = np.ones(len(spectrum), dtype=bool)
        r = 5
        mask[max(0, peak_idx-r):min(len(spectrum), peak_idx+r+1)] = False

        if mask.sum() > 0:
            background = spectrum[mask].mean()
        else:
            background = 0.1

        if peak_val > 0:
            prominence = (peak_val - background) / peak_val
            confidence = min(0.95, max(0.3, prominence))
        else:
            confidence = 0.3

        return confidence

    def angle_to_beam_index(self, azimuth: float, elevation: float, num_beams_h: int, num_beams_v: int) -> int:
        """Convert angles to beam codebook index"""
        az_range = self.config.azimuth_range[1] - self.config.azimuth_range[0]
        el_range = self.config.elevation_range[1] - self.config.elevation_range[0]

        az_norm = (azimuth - self.config.azimuth_range[0]) / az_range
        el_norm = (elevation - self.config.elevation_range[0]) / el_range

        az_norm = np.clip(az_norm, 0, 1)
        el_norm = np.clip(el_norm, 0, 1)

        h_idx = int(az_norm * num_beams_h) % num_beams_h
        v_idx = int(el_norm * num_beams_v) % num_beams_v

        return h_idx * num_beams_v + v_idx

    def _update_stats(self, result: AngleEstimate):
        """Update estimation statistics"""
        self.stats["estimates"] += 1
        n = self.stats["estimates"]
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (n - 1) / n + result.confidence / n
        )

        method = result.method
        if method not in self.stats["method_usage"]:
            self.stats["method_usage"][method] = 0
        self.stats["method_usage"][method] += 1

    def get_statistics(self) -> Dict:
        """Get estimation statistics"""
        return self.stats.copy()

    def clear_cache(self):
        """Clear steering vector cache"""
        self.steering_vector_cached.cache_clear()
