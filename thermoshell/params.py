import dataclasses
import typing
import numpy as np

@dataclasses.dataclass(frozen=True)
class GeomParams:
    mesh_id: int = 2
    
    outer_radius: float = 0.05
    
    pattern_center_x: float = 0.0
    pattern_center_y: float = 0.0
    stripe_width: float = 0.0059
    stripe_length: float = 0.0441
    stripe_radius: float = 0.0029
    
    delta_shape: float = 0.02787
    n_spokes: int = 6
    star_radius: float = 0.044595
    star_thickness: float = 0.002787
    beam_thickness: float = 0.002787
    
    l_k: float = 0.0040   
    h_soft: float = 0.3e-3 
    h_hard: float = 0.6e-3 


@dataclasses.dataclass(frozen=True)
class MaterialParams:
    youngs_soft: float = 1.0e6
    youngs_hard: float = 3.0e6
    youngs_ratio: float = 1.0 

    thermal_strain_mag: float = -0.3
    thermal_strain_min: float = 0.0
    thermal_strain_max: float = -0.3
    fluctuation_percentage: float = 2.0
    
    factor_ks: float = 10.0
    factor_kb: float = 1.0
    
    nodal_mass: float = 1.0e-7
    
    coupling_beta: float = 1.0


@dataclasses.dataclass(frozen=True)
class SimParams:
    total_time: float = 1.0
    dt_initial: float = 0.05
    dt_min: float = 1.0e-8
    dt_max: float = 0.1

    tolerance: float = 1.0e-5  
    max_newton_iters: int = 20
    
    use_gravity: bool = True   
    use_fluctuations: bool = False 
    
    gravity_vector: np.ndarray = np.array([0.0, 0.0, 9.81]) 
    
    num_records: int = 3 
    relax_time: float = 1.0
    relax_steps: int = 10
