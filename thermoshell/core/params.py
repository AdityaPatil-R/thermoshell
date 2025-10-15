import dataclasses
import typing
import numpy as np

# A common utility type for clarity
Array = typing.NewType('Array', np.ndarray)


@dataclasses.dataclass(frozen=True)
class GeomParams:
    """Parameters defining the geometry, mesh properties, and patterning."""
    mesh_id: int = 2
    
    # Outer mesh size constants (used primarily for pattern definition boundaries)
    outer_R: float = 0.05
    
    # Pattern parameters (for iMesh=1, 3)
    stripe_width: float = 0.0059
    stripe_length: float = 0.0707
    stripe_r: float = 0.0006
    
    # Pattern parameters (for iMesh=2 - Peanut/Canoe shape)
    delta_shape: float = 0.02787
    n_spokes: int = 6
    
    # Calculated based on mesh load, default is centered
    pattern_center_x: float = 0.0
    pattern_center_y: float = 0.0


@dataclasses.dataclass(frozen=True)
class MaterialParams:
    """Parameters defining material properties and constitutive behavior."""
    
    # Young's Moduli (Pa) - Hard layer simulates PLA, Soft layer simulates Shrinky Dink
    youngs_soft: float = 1.0e6
    youngs_hard: float = 3.0e6
    
    # Thermal Actuation Magnitude (max strain)
    thermal_strain_mag: float = -0.3
    
    # Effective Mass (per DOF)
    nodal_mass: float = 1.0e-7
    
    # Stiffness scaling factors
    factor_ks: float = 10.0
    factor_kb: float = 1.0
    
    # Axial-Bending Coupling Coefficient (beta in the energy equation)
    coupling_beta: float = 1.0


@dataclasses.dataclass(frozen=True)
class SimParams:
    """Parameters controlling the solver, time stepping, and output flags."""
    
    # Time Stepping
    total_time: float = 1.0        # Total simulated load time
    dt_initial: float = 0.05       # Initial timestep size for ramp-up
    dt_min: float = 1.0e-8         # Minimum allowable timestep (for stability)
    dt_max: float = 0.1            # Maximum allowable timestep (for efficiency)
    
    # Solver
    tolerance: float = 1.0e-5      # Relative error tolerance for Newton-Raphson
    max_newton_iters: int = 20     # Max iterations per timestep
    
    # External Forces
    use_gravity: bool = True
    # Note: Vector is [gx, gy, gz]. Sign is inverse of physical world due to Z-axis convention.
    gravity_vector: Array = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 9.81]))
    
    # Gravity Relaxation Phase
    relax_time: float = 1.0        # Time over which gravity is ramped down after thermal actuation
    relax_steps: int = 10          # Number of steps during relaxation
    
    # Flags and Output
    use_fluctuations: bool = False # Enable sinusoidal boundary fluctuation in thermal strain
    num_records: int = 3           # Max records to save in history arrays (Initial, Actuated, Relaxed)
    
    # Console Flags (Set via CLI args)
    do_print: bool = False
    do_plot: bool = False
    do_test: bool = False


@dataclasses.dataclass(frozen=True)
class DerivedParams:
    """
    Holds material properties calculated once based on GeomParams and MaterialParams,
    such as individual layer thicknesses and derived effective stiffness constants.
    """
    
    # Geometry/Layer Thicknesses (determined from mesh ID lookup)
    thickness_soft: typing.Optional[float] = None
    thickness_hard: typing.Optional[float] = None
    ref_length_k: typing.Optional[float] = None
    
    # Effective Stiffness Constants (per unit width, or specific to hinge geometry)
    ks1: typing.Optional[float] = None
    ks2: typing.Optional[float] = None
    ks12: typing.Optional[float] = None
    kb1: typing.Optional[float] = None
    kb12: typing.Optional[float] = None
