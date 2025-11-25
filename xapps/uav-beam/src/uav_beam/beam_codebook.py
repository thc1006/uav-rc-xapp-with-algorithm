"""
Beam Codebook Management for 5G NR mmWave Systems

Implements beam codebook generation and management per 3GPP TS 38.214:
- DFT-based codebook for ULA/UPA arrays
- Hierarchical multi-resolution codebook (Type I/II)
- Beam pattern computation
- Beam gain modeling

References:
- 3GPP TS 38.214: Physical layer procedures for data
- 3GPP TS 38.211: Physical channels and modulation
- IEEE 802.11ad/ay: mmWave beamforming standards
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CodebookType(Enum):
    """Beam codebook types per 3GPP TS 38.214"""
    TYPE_I_SINGLE_PANEL = "type_i_single"    # Single panel, wideband
    TYPE_I_MULTI_PANEL = "type_i_multi"      # Multi-panel
    TYPE_II = "type_ii"                       # High resolution
    DFT = "dft"                               # Standard DFT codebook
    HIERARCHICAL = "hierarchical"             # Multi-level hierarchical


class ArrayType(Enum):
    """Antenna array geometry"""
    ULA = "ula"           # Uniform Linear Array
    UPA = "upa"           # Uniform Planar Array (2D)
    UCA = "uca"           # Uniform Circular Array


@dataclass
class CodebookConfig:
    """Beam codebook configuration parameters"""
    # Array geometry
    array_type: ArrayType = ArrayType.UPA
    num_elements_h: int = 8                 # Horizontal antenna elements (N1)
    num_elements_v: int = 8                 # Vertical antenna elements (N2)
    element_spacing_h: float = 0.5          # Horizontal spacing (wavelengths)
    element_spacing_v: float = 0.5          # Vertical spacing (wavelengths)

    # Codebook parameters per 3GPP TS 38.214 Table 5.2.2.2.1-2
    num_beams_h: int = 16                   # O1: Horizontal oversampling
    num_beams_v: int = 8                    # O2: Vertical oversampling

    # Hierarchical codebook levels
    num_levels: int = 3                     # Number of hierarchy levels

    # Polarization
    num_polarizations: int = 2              # Single or dual polarization

    # Frequency parameters
    carrier_frequency_ghz: float = 28.0     # FR2 carrier frequency

    # Beam characteristics
    beam_width_3db_h: float = 10.0          # Horizontal 3dB beamwidth (degrees)
    beam_width_3db_v: float = 15.0          # Vertical 3dB beamwidth (degrees)

    @property
    def total_elements(self) -> int:
        """Total number of antenna elements"""
        return self.num_elements_h * self.num_elements_v * self.num_polarizations

    @property
    def total_beams(self) -> int:
        """Total number of beams in codebook"""
        return self.num_beams_h * self.num_beams_v

    @property
    def wavelength_m(self) -> float:
        """Wavelength in meters"""
        return 3e8 / (self.carrier_frequency_ghz * 1e9)


@dataclass
class BeamPattern:
    """Beam radiation pattern"""
    beam_id: int
    azimuth_center_rad: float
    elevation_center_rad: float
    azimuth_3db_rad: float
    elevation_3db_rad: float
    peak_gain_dbi: float
    steering_vector: np.ndarray
    pattern_2d: Optional[np.ndarray] = None  # 2D gain pattern


@dataclass
class HierarchicalLevel:
    """Single level of hierarchical codebook"""
    level: int
    num_beams_h: int
    num_beams_v: int
    beam_width_h_rad: float
    beam_width_v_rad: float
    codebook: np.ndarray                    # Shape: (num_beams, num_elements)
    beam_directions: List[Tuple[float, float]]  # (azimuth, elevation) pairs


class BeamCodebook:
    """
    Beam Codebook Manager for 5G NR mmWave

    Generates and manages beam codebooks for:
    - SSB transmission (P1 procedure)
    - CSI-RS based beam refinement (P2 procedure)
    - Beam tracking (P3 procedure)

    Supports 3GPP Type I and Type II codebooks as well as
    hierarchical multi-resolution codebooks for fast beam search.
    """

    def __init__(self, config: Optional[CodebookConfig] = None):
        self.config = config or CodebookConfig()

        # Pre-computed codebooks
        self._dft_codebook: Optional[np.ndarray] = None
        self._hierarchical_codebook: Optional[List[HierarchicalLevel]] = None
        self._beam_patterns: Dict[int, BeamPattern] = {}

        # Angular grids for pattern computation
        self._az_grid = np.linspace(-np.pi/2, np.pi/2, 181)
        self._el_grid = np.linspace(-np.pi/4, np.pi/4, 91)

        # Initialize codebooks
        self._generate_dft_codebook()
        self._generate_hierarchical_codebook()
        self._compute_beam_patterns()

        logger.info(
            f"BeamCodebook initialized: {self.config.num_elements_h}x"
            f"{self.config.num_elements_v} UPA, {self.config.total_beams} beams"
        )

    # =========================================================================
    # Steering Vector Computation
    # =========================================================================

    def steering_vector_ula(
        self,
        angle_rad: float,
        num_elements: int,
        spacing: float = 0.5
    ) -> np.ndarray:
        """
        Compute steering vector for Uniform Linear Array

        Args:
            angle_rad: Angle of arrival/departure in radians
            num_elements: Number of array elements
            spacing: Element spacing in wavelengths

        Returns:
            Steering vector of shape (num_elements,)
        """
        n = np.arange(num_elements)
        return np.exp(1j * 2 * np.pi * spacing * n * np.sin(angle_rad)) / np.sqrt(num_elements)

    def steering_vector_upa(
        self,
        azimuth_rad: float,
        elevation_rad: float
    ) -> np.ndarray:
        """
        Compute steering vector for Uniform Planar Array

        Per 3GPP TS 38.214, the array response for a 2D UPA:
        a(theta, phi) = a_h(theta, phi) kronecker a_v(phi)

        where:
        - theta: azimuth angle
        - phi: elevation angle

        Args:
            azimuth_rad: Azimuth angle in radians
            elevation_rad: Elevation angle in radians

        Returns:
            Steering vector of shape (num_elements,)
        """
        M = self.config.num_elements_h
        N = self.config.num_elements_v
        d_h = self.config.element_spacing_h
        d_v = self.config.element_spacing_v

        # Spatial frequencies
        u = np.sin(azimuth_rad) * np.cos(elevation_rad)  # Horizontal
        v = np.sin(elevation_rad)                         # Vertical

        # Horizontal steering vector
        m = np.arange(M)
        a_h = np.exp(1j * 2 * np.pi * d_h * m * u)

        # Vertical steering vector
        n = np.arange(N)
        a_v = np.exp(1j * 2 * np.pi * d_v * n * v)

        # Kronecker product for 2D array
        sv = np.kron(a_h, a_v)

        return sv / np.linalg.norm(sv)

    def steering_vector_dual_pol(
        self,
        azimuth_rad: float,
        elevation_rad: float,
        polarization_phase: float = 0.0
    ) -> np.ndarray:
        """
        Compute steering vector for dual-polarized UPA

        Args:
            azimuth_rad: Azimuth angle in radians
            elevation_rad: Elevation angle in radians
            polarization_phase: Phase between polarizations (radians)

        Returns:
            Steering vector including both polarizations
        """
        # Single polarization steering vector
        sv_single = self.steering_vector_upa(azimuth_rad, elevation_rad)

        if self.config.num_polarizations == 1:
            return sv_single

        # Dual polarization: [sv_pol1; sv_pol2 * exp(j*phase)]
        sv_pol1 = sv_single
        sv_pol2 = sv_single * np.exp(1j * polarization_phase)

        return np.concatenate([sv_pol1, sv_pol2]) / np.sqrt(2)

    # =========================================================================
    # DFT Codebook Generation (3GPP Type I Single Panel)
    # =========================================================================

    def _generate_dft_codebook(self):
        """
        Generate DFT-based codebook per 3GPP TS 38.214 Section 5.2.2.2.1

        For Type I Single-Panel, the precoder is:
        W = W1 * W2

        where W1 contains wideband beam directions and W2 contains
        co-phasing across polarizations.
        """
        O1 = self.config.num_beams_h  # Horizontal oversampling factor
        O2 = self.config.num_beams_v  # Vertical oversampling factor
        N1 = self.config.num_elements_h
        N2 = self.config.num_elements_v

        total_beams = O1 * O2
        total_elements = N1 * N2

        self._dft_codebook = np.zeros((total_beams, total_elements), dtype=complex)

        beam_idx = 0
        for i1 in range(O1):
            for i2 in range(O2):
                # Beam direction angles per 3GPP formula
                # Map oversampled indices to angles
                azimuth = np.arcsin((2 * i1 / O1 - 1))      # [-pi/2, pi/2]
                elevation = np.arcsin((2 * i2 / O2 - 1) * 0.5)  # [-pi/4, pi/4]

                # Compute steering vector
                self._dft_codebook[beam_idx, :] = self.steering_vector_upa(
                    azimuth, elevation
                )

                beam_idx += 1

        logger.debug(f"Generated DFT codebook: {total_beams} beams")

    # =========================================================================
    # Hierarchical Codebook Generation
    # =========================================================================

    def _generate_hierarchical_codebook(self):
        """
        Generate multi-level hierarchical codebook

        Level 0: Wide beams covering entire sector (few beams, wide coverage)
        Level 1: Medium beams
        Level 2+: Narrow beams (many beams, narrow coverage)

        This enables fast beam acquisition through hierarchical search:
        1. First find best wide beam
        2. Then refine with narrower beams within that sector
        """
        self._hierarchical_codebook = []

        base_beams_h = 2
        base_beams_v = 2

        for level in range(self.config.num_levels):
            # Number of beams increases with level
            num_beams_h = base_beams_h * (2 ** level)
            num_beams_v = base_beams_v * (2 ** level)

            # Clamp to maximum resolution
            num_beams_h = min(num_beams_h, self.config.num_beams_h)
            num_beams_v = min(num_beams_v, self.config.num_beams_v)

            # Beam width decreases with level (wider beams at lower levels)
            beam_width_h = np.pi / num_beams_h
            beam_width_v = np.pi / (2 * num_beams_v)

            # Generate codebook for this level
            level_codebook = np.zeros(
                (num_beams_h * num_beams_v, self.config.num_elements_h * self.config.num_elements_v),
                dtype=complex
            )
            beam_directions = []

            beam_idx = 0
            for i_h in range(num_beams_h):
                for i_v in range(num_beams_v):
                    # Beam center direction
                    azimuth = -np.pi/2 + (i_h + 0.5) * np.pi / num_beams_h
                    elevation = -np.pi/4 + (i_v + 0.5) * np.pi / (2 * num_beams_v)

                    # For wider beams, use weighted sum of neighboring steering vectors
                    if level < self.config.num_levels - 1:
                        sv = self._compute_wide_beam_weights(
                            azimuth, elevation, beam_width_h, beam_width_v
                        )
                    else:
                        sv = self.steering_vector_upa(azimuth, elevation)

                    level_codebook[beam_idx, :] = sv
                    beam_directions.append((azimuth, elevation))
                    beam_idx += 1

            self._hierarchical_codebook.append(HierarchicalLevel(
                level=level,
                num_beams_h=num_beams_h,
                num_beams_v=num_beams_v,
                beam_width_h_rad=beam_width_h,
                beam_width_v_rad=beam_width_v,
                codebook=level_codebook,
                beam_directions=beam_directions
            ))

        logger.debug(f"Generated hierarchical codebook with {self.config.num_levels} levels")

    def _compute_wide_beam_weights(
        self,
        center_az: float,
        center_el: float,
        width_az: float,
        width_el: float,
        num_samples: int = 5
    ) -> np.ndarray:
        """
        Compute beamforming weights for wide beam by combining multiple directions

        This creates a wider beam pattern by averaging steering vectors
        over the beam coverage area.
        """
        num_elements = self.config.num_elements_h * self.config.num_elements_v
        combined_sv = np.zeros(num_elements, dtype=complex)

        # Sample points within beam coverage
        az_samples = np.linspace(
            center_az - width_az/2,
            center_az + width_az/2,
            num_samples
        )
        el_samples = np.linspace(
            center_el - width_el/2,
            center_el + width_el/2,
            num_samples
        )

        # Weight samples (Gaussian-like weighting)
        total_weight = 0
        for az in az_samples:
            for el in el_samples:
                # Gaussian weight based on distance from center
                weight = np.exp(-(
                    ((az - center_az) / width_az) ** 2 +
                    ((el - center_el) / width_el) ** 2
                ))
                combined_sv += weight * self.steering_vector_upa(az, el)
                total_weight += weight

        return combined_sv / np.linalg.norm(combined_sv)

    # =========================================================================
    # Beam Pattern Computation
    # =========================================================================

    def _compute_beam_patterns(self):
        """Compute and cache beam patterns for all beams"""
        for beam_id in range(self.config.total_beams):
            self._beam_patterns[beam_id] = self._compute_single_beam_pattern(beam_id)

    def _compute_single_beam_pattern(self, beam_id: int) -> BeamPattern:
        """
        Compute beam radiation pattern

        Beam pattern = |w^H * a(theta, phi)|^2

        where w is the beamforming weight and a is the array response.
        """
        if self._dft_codebook is None:
            self._generate_dft_codebook()

        w = self._dft_codebook[beam_id, :]

        # Beam center direction
        i_h = beam_id // self.config.num_beams_v
        i_v = beam_id % self.config.num_beams_v
        center_az = np.arcsin((2 * i_h / self.config.num_beams_h - 1))
        center_el = np.arcsin((2 * i_v / self.config.num_beams_v - 1) * 0.5)

        # Compute 2D pattern
        pattern = np.zeros((len(self._az_grid), len(self._el_grid)))

        for i_az, az in enumerate(self._az_grid):
            for i_el, el in enumerate(self._el_grid):
                a = self.steering_vector_upa(az, el)
                gain = np.abs(np.vdot(w, a)) ** 2
                pattern[i_az, i_el] = gain

        # Normalize to peak = 1
        peak_gain = pattern.max()
        if peak_gain > 0:
            pattern /= peak_gain

        # Find 3dB beamwidth
        az_3db, el_3db = self._compute_3db_beamwidth(pattern, center_az, center_el)

        # Compute peak gain in dBi (assuming ideal aperture)
        directivity = 4 * np.pi * self.config.num_elements_h * self.config.num_elements_v
        peak_gain_dbi = 10 * np.log10(directivity)

        return BeamPattern(
            beam_id=beam_id,
            azimuth_center_rad=center_az,
            elevation_center_rad=center_el,
            azimuth_3db_rad=az_3db,
            elevation_3db_rad=el_3db,
            peak_gain_dbi=peak_gain_dbi,
            steering_vector=w,
            pattern_2d=pattern
        )

    def _compute_3db_beamwidth(
        self,
        pattern: np.ndarray,
        center_az: float,
        center_el: float
    ) -> Tuple[float, float]:
        """Compute 3dB beamwidth from pattern"""
        # Find center indices
        center_az_idx = np.argmin(np.abs(self._az_grid - center_az))
        center_el_idx = np.argmin(np.abs(self._el_grid - center_el))

        # Azimuth cut (at center elevation)
        az_cut = pattern[:, center_el_idx]
        az_3db = self._find_3db_width(az_cut, center_az_idx, self._az_grid)

        # Elevation cut (at center azimuth)
        el_cut = pattern[center_az_idx, :]
        el_3db = self._find_3db_width(el_cut, center_el_idx, self._el_grid)

        return az_3db, el_3db

    def _find_3db_width(
        self,
        cut: np.ndarray,
        center_idx: int,
        angle_grid: np.ndarray
    ) -> float:
        """Find 3dB width from 1D pattern cut"""
        threshold = 0.5  # -3dB = half power

        # Find left edge
        left_idx = center_idx
        while left_idx > 0 and cut[left_idx] > threshold:
            left_idx -= 1

        # Find right edge
        right_idx = center_idx
        while right_idx < len(cut) - 1 and cut[right_idx] > threshold:
            right_idx += 1

        width = angle_grid[right_idx] - angle_grid[left_idx]
        return width

    # =========================================================================
    # Beam Gain Modeling
    # =========================================================================

    def compute_beam_gain(
        self,
        beam_id: int,
        azimuth_rad: float,
        elevation_rad: float
    ) -> float:
        """
        Compute beam gain for given direction

        Args:
            beam_id: Beam index
            azimuth_rad: Target azimuth angle
            elevation_rad: Target elevation angle

        Returns:
            Beam gain in dB
        """
        if self._dft_codebook is None:
            self._generate_dft_codebook()

        w = self._dft_codebook[beam_id, :]
        a = self.steering_vector_upa(azimuth_rad, elevation_rad)

        # Array gain
        array_gain = np.abs(np.vdot(w, a)) ** 2

        # Convert to dB and add element gain
        gain_db = 10 * np.log10(array_gain + 1e-10)

        # Add antenna element gain (typically 5-8 dBi for patch antenna)
        element_gain_dbi = 5.0

        return gain_db + element_gain_dbi

    def compute_path_loss(
        self,
        distance_m: float,
        frequency_ghz: Optional[float] = None,
        los: bool = True
    ) -> float:
        """
        Compute path loss per 3GPP TR 38.901

        Uses Urban Micro (UMi) Street Canyon model for mmWave

        Args:
            distance_m: 3D distance in meters
            frequency_ghz: Carrier frequency (default: config value)
            los: Line-of-sight condition

        Returns:
            Path loss in dB
        """
        f_c = frequency_ghz or self.config.carrier_frequency_ghz
        d_3d = max(distance_m, 1.0)  # Minimum 1m

        if los:
            # LOS path loss (3GPP UMi-Street Canyon)
            pl = 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(f_c)
        else:
            # NLOS path loss
            pl = 32.4 + 31.9 * np.log10(d_3d) + 20.0 * np.log10(f_c)

        return pl

    def compute_received_power(
        self,
        tx_power_dbm: float,
        beam_id: int,
        target_azimuth_rad: float,
        target_elevation_rad: float,
        distance_m: float,
        los: bool = True
    ) -> float:
        """
        Compute received power including beam gain and path loss

        P_rx = P_tx + G_tx + G_rx - PL

        Args:
            tx_power_dbm: Transmit power in dBm
            beam_id: Transmit beam index
            target_azimuth_rad: Target direction azimuth
            target_elevation_rad: Target direction elevation
            distance_m: Distance to target
            los: Line-of-sight condition

        Returns:
            Received power in dBm (L1-RSRP)
        """
        # Beam gain at target direction
        beam_gain = self.compute_beam_gain(
            beam_id, target_azimuth_rad, target_elevation_rad
        )

        # Path loss
        path_loss = self.compute_path_loss(distance_m, los=los)

        # Assume omnidirectional receive antenna (0 dBi)
        rx_gain = 0.0

        # Link budget
        rx_power = tx_power_dbm + beam_gain + rx_gain - path_loss

        return rx_power

    # =========================================================================
    # Codebook Access Methods
    # =========================================================================

    def get_beam_steering_vector(self, beam_id: int) -> np.ndarray:
        """Get steering vector for given beam"""
        if self._dft_codebook is None:
            self._generate_dft_codebook()
        return self._dft_codebook[beam_id, :].copy()

    def get_dft_codebook(self) -> np.ndarray:
        """Get full DFT codebook matrix"""
        if self._dft_codebook is None:
            self._generate_dft_codebook()
        return self._dft_codebook.copy()

    def get_hierarchical_level(self, level: int) -> Optional[HierarchicalLevel]:
        """Get codebook for specific hierarchy level"""
        if self._hierarchical_codebook is None:
            self._generate_hierarchical_codebook()

        if 0 <= level < len(self._hierarchical_codebook):
            return self._hierarchical_codebook[level]
        return None

    def get_beam_pattern(self, beam_id: int) -> Optional[BeamPattern]:
        """Get pre-computed beam pattern"""
        return self._beam_patterns.get(beam_id)

    def get_beam_direction(self, beam_id: int) -> Tuple[float, float]:
        """
        Get beam center direction

        Returns:
            (azimuth_rad, elevation_rad)
        """
        i_h = beam_id // self.config.num_beams_v
        i_v = beam_id % self.config.num_beams_v

        azimuth = np.arcsin((2 * i_h / self.config.num_beams_h - 1))
        elevation = np.arcsin((2 * i_v / self.config.num_beams_v - 1) * 0.5)

        return azimuth, elevation

    def angle_to_beam_id(
        self,
        azimuth_rad: float,
        elevation_rad: float
    ) -> int:
        """
        Map angles to nearest beam index

        Args:
            azimuth_rad: Azimuth angle in radians
            elevation_rad: Elevation angle in radians

        Returns:
            Nearest beam ID
        """
        # Clamp angles
        azimuth_rad = np.clip(azimuth_rad, -np.pi/2, np.pi/2)
        elevation_rad = np.clip(elevation_rad, -np.pi/4, np.pi/4)

        # Map to indices
        i_h = int((np.sin(azimuth_rad) + 1) / 2 * self.config.num_beams_h)
        i_v = int((np.sin(elevation_rad) / 0.5 + 1) / 2 * self.config.num_beams_v)

        # Clamp to valid range
        i_h = np.clip(i_h, 0, self.config.num_beams_h - 1)
        i_v = np.clip(i_v, 0, self.config.num_beams_v - 1)

        return i_h * self.config.num_beams_v + i_v

    def get_neighboring_beams(
        self,
        beam_id: int,
        neighborhood_size: int = 1
    ) -> List[int]:
        """
        Get IDs of neighboring beams

        Args:
            beam_id: Center beam ID
            neighborhood_size: Number of neighbors in each direction

        Returns:
            List of neighboring beam IDs
        """
        i_h = beam_id // self.config.num_beams_v
        i_v = beam_id % self.config.num_beams_v

        neighbors = []

        for di in range(-neighborhood_size, neighborhood_size + 1):
            for dj in range(-neighborhood_size, neighborhood_size + 1):
                if di == 0 and dj == 0:
                    continue

                ni = i_h + di
                nj = i_v + dj

                if 0 <= ni < self.config.num_beams_h and 0 <= nj < self.config.num_beams_v:
                    neighbors.append(ni * self.config.num_beams_v + nj)

        return neighbors

    # =========================================================================
    # SSB Beam Configuration (for P1 procedure)
    # =========================================================================

    def get_ssb_beam_set(self, num_ssb_beams: int = 8) -> List[int]:
        """
        Get beam IDs for SSB transmission

        SSB beams are typically wide beams covering the entire sector.
        Per 3GPP, up to 64 SSB beams can be configured for FR2.

        Args:
            num_ssb_beams: Number of SSB beams (4, 8, or 64 per 3GPP)

        Returns:
            List of beam IDs for SSB transmission
        """
        # Select evenly spaced beams from hierarchy level 0 or 1
        if num_ssb_beams <= 4:
            level = 0
        elif num_ssb_beams <= 16:
            level = 1
        else:
            level = 2

        hier_level = self.get_hierarchical_level(level)
        if hier_level is None:
            # Fall back to DFT codebook
            step = max(1, self.config.total_beams // num_ssb_beams)
            return list(range(0, self.config.total_beams, step))[:num_ssb_beams]

        total_level_beams = hier_level.num_beams_h * hier_level.num_beams_v
        step = max(1, total_level_beams // num_ssb_beams)
        return list(range(0, total_level_beams, step))[:num_ssb_beams]

    def get_csirs_beam_set(
        self,
        ssb_beam_id: int,
        num_csirs_beams: int = 8
    ) -> List[int]:
        """
        Get CSI-RS beam set for beam refinement within SSB beam coverage

        Args:
            ssb_beam_id: Selected SSB beam
            num_csirs_beams: Number of CSI-RS beams for refinement

        Returns:
            List of narrow beam IDs within SSB beam coverage
        """
        # Get SSB beam direction
        ssb_az, ssb_el = self.get_beam_direction(ssb_beam_id)

        # Find narrow beams in fine codebook near this direction
        candidates = []
        for beam_id in range(self.config.total_beams):
            az, el = self.get_beam_direction(beam_id)

            # Check if within SSB beam coverage (approximate)
            ssb_width_az = np.pi / 4  # Approximate SSB beam width
            ssb_width_el = np.pi / 8

            if (abs(az - ssb_az) < ssb_width_az and
                abs(el - ssb_el) < ssb_width_el):
                candidates.append(beam_id)

        # Select evenly distributed subset
        if len(candidates) <= num_csirs_beams:
            return candidates

        step = len(candidates) // num_csirs_beams
        return candidates[::step][:num_csirs_beams]

    def get_statistics(self) -> Dict:
        """Get codebook statistics"""
        return {
            "array_type": self.config.array_type.value,
            "num_elements": self.config.total_elements,
            "num_beams": self.config.total_beams,
            "num_hierarchy_levels": len(self._hierarchical_codebook) if self._hierarchical_codebook else 0,
            "carrier_frequency_ghz": self.config.carrier_frequency_ghz,
        }
