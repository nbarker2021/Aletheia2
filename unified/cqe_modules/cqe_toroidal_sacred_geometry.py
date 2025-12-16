from pathlib import Path
#!/usr/bin/env python3
"""
CQE Toroidal Sacred Geometry Module
Exposes relationships between forces and sacred geometry through toroidal shell rotations
Integrates Carlson's rotational principles with E₈ mathematics in toroidal framework
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ToroidalRotationType(Enum):
    """Types of rotations around toroidal shell"""
    POLOIDAL = "POLOIDAL"          # Around minor circumference (inward/9-pattern)
    TOROIDAL = "TOROIDAL"          # Around major circumference (outward/6-pattern)
    MERIDIONAL = "MERIDIONAL"      # Through torus center (creative/3-pattern)
    HELICAL = "HELICAL"            # Spiral combination (transformative)

class ForceType(Enum):
    """Classification of forces by sacred geometry patterns"""
    GRAVITATIONAL = "GRAVITATIONAL"    # Inward/convergent (9-pattern)
    ELECTROMAGNETIC = "ELECTROMAGNETIC" # Outward/divergent (6-pattern)
    NUCLEAR_STRONG = "NUCLEAR_STRONG"   # Creative/binding (3-pattern)
    NUCLEAR_WEAK = "NUCLEAR_WEAK"      # Transformative/decay (other patterns)

@dataclass
class ToroidalCoordinate:
    """Toroidal coordinate system (R, θ, φ) with sacred geometry properties"""
    R: float          # Major radius (distance from torus center)
    theta: float      # Poloidal angle (around minor circumference)
    phi: float        # Toroidal angle (around major circumference)
    
    # Sacred geometry properties
    digital_root: int
    rotational_pattern: str
    sacred_frequency: float
    force_classification: ForceType
    
    def to_cartesian(self, r: float = 1.0) -> Tuple[float, float, float]:
        """Convert toroidal coordinates to Cartesian with minor radius r"""
        x = (self.R + r * math.cos(self.theta)) * math.cos(self.phi)
        y = (self.R + r * math.cos(self.theta)) * math.sin(self.phi)
        z = r * math.sin(self.theta)
        return (x, y, z)
    
    def calculate_rotational_energy(self) -> float:
        """Calculate rotational energy based on sacred geometry"""
        # Base energy from toroidal position
        base_energy = self.R * (math.sin(self.theta)**2 + math.cos(self.phi)**2)
        
        # Sacred geometry modulation
        if self.digital_root == 9:  # Inward/convergent
            return base_energy * (432.0 / 440.0)  # 432 Hz resonance
        elif self.digital_root == 6:  # Outward/divergent
            return base_energy * (528.0 / 440.0)  # 528 Hz resonance
        elif self.digital_root == 3:  # Creative/generative
            return base_energy * (396.0 / 440.0)  # 396 Hz resonance
        else:  # Transformative
            return base_energy * (741.0 / 440.0)  # 741 Hz resonance

class ToroidalSacredGeometry:
    """Core toroidal sacred geometry engine"""
    
    def __init__(self, major_radius: float = 3.0, minor_radius: float = 1.0):
        self.major_radius = major_radius  # R (3 -> creative seed)
        self.minor_radius = minor_radius  # r (1 -> unity)
        
        # Sacred ratios
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.silver_ratio = 1 + math.sqrt(2)
        
        # Sacred frequencies (Hz)
        self.sacred_frequencies = {
            9: 432.0,   # Inward/completion
            6: 528.0,   # Outward/creation
            3: 396.0,   # Creative/liberation
            1: 741.0,   # Transformative/expression
            2: 852.0,   # Transformative/intuition
            4: 963.0,   # Inward/connection
            5: 174.0,   # Transformative/foundation
            7: 285.0,   # Transformative/change
            8: 639.0    # Transformative/relationships
        }
        
        # E₈ integration parameters
        self.e8_embedding_scale = 1.0 / math.sqrt(8)
        
    def calculate_digital_root(self, n: float) -> int:
        """Calculate digital root using Carlson's method"""
        n = abs(int(n * 1000))  # Scale for floating point
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n if n > 0 else 9
    
    def classify_rotational_pattern(self, digital_root: int) -> str:
        """Classify by Carlson's rotational patterns"""
        if digital_root == 9:
            return "INWARD_ROTATIONAL"
        elif digital_root == 6:
            return "OUTWARD_ROTATIONAL"
        elif digital_root == 3:
            return "CREATIVE_SEED"
        else:
            return "TRANSFORMATIVE_CYCLE"
    
    def classify_force_type(self, digital_root: int, rotational_energy: float) -> ForceType:
        """Classify force type based on sacred geometry and energy"""
        if digital_root == 9 and rotational_energy < 1.0:
            return ForceType.GRAVITATIONAL
        elif digital_root == 6 and rotational_energy > 1.0:
            return ForceType.ELECTROMAGNETIC
        elif digital_root == 3:
            return ForceType.NUCLEAR_STRONG
        else:
            return ForceType.NUCLEAR_WEAK
    
    def create_toroidal_coordinate(self, R: float, theta: float, phi: float) -> ToroidalCoordinate:
        """Create toroidal coordinate with sacred geometry properties"""
        
        # Calculate digital root from position
        position_value = R * 1000 + theta * 100 + phi * 10
        digital_root = self.calculate_digital_root(position_value)
        
        # Classify rotational pattern
        rotational_pattern = self.classify_rotational_pattern(digital_root)
        
        # Get sacred frequency
        sacred_frequency = self.sacred_frequencies.get(digital_root, 440.0)
        
        # Create coordinate
        coord = ToroidalCoordinate(
            R=R, theta=theta, phi=phi,
            digital_root=digital_root,
            rotational_pattern=rotational_pattern,
            sacred_frequency=sacred_frequency,
            force_classification=ForceType.GRAVITATIONAL  # Will be updated
        )
        
        # Calculate rotational energy and classify force
        rotational_energy = coord.calculate_rotational_energy()
        coord.force_classification = self.classify_force_type(digital_root, rotational_energy)
        
        return coord
    
    def generate_toroidal_shell(self, theta_points: int = 36, phi_points: int = 72) -> List[ToroidalCoordinate]:
        """Generate complete toroidal shell with sacred geometry classification"""
        
        shell_points = []
        
        for i in range(theta_points):
            theta = 2 * math.pi * i / theta_points
            
            for j in range(phi_points):
                phi = 2 * math.pi * j / phi_points
                
                # Use golden ratio for major radius variation
                R = self.major_radius * (1 + 0.1 * math.sin(theta * self.golden_ratio))
                
                coord = self.create_toroidal_coordinate(R, theta, phi)
                shell_points.append(coord)
        
        return shell_points
    
    def analyze_rotational_forces(self, shell_points: List[ToroidalCoordinate]) -> Dict[str, Any]:
        """Analyze rotational forces across toroidal shell"""
        
        force_analysis = {
            'total_points': len(shell_points),
            'pattern_distribution': {},
            'force_distribution': {},
            'energy_statistics': {},
            'sacred_frequency_map': {}
        }
        
        # Analyze pattern distribution
        pattern_counts = {}
        force_counts = {}
        energies = []
        frequency_map = {}
        
        for coord in shell_points:
            # Pattern distribution
            pattern = coord.rotational_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Force distribution
            force = coord.force_classification.value
            force_counts[force] = force_counts.get(force, 0) + 1
            
            # Energy statistics
            energy = coord.calculate_rotational_energy()
            energies.append(energy)
            
            # Sacred frequency mapping
            freq = coord.sacred_frequency
            if freq not in frequency_map:
                frequency_map[freq] = []
            frequency_map[freq].append((coord.theta, coord.phi))
        
        force_analysis['pattern_distribution'] = pattern_counts
        force_analysis['force_distribution'] = force_counts
        force_analysis['energy_statistics'] = {
            'mean': np.mean(energies),
            'std': np.std(energies),
            'min': np.min(energies),
            'max': np.max(energies)
        }
        force_analysis['sacred_frequency_map'] = frequency_map
        
        return force_analysis
    
    def embed_toroidal_in_e8(self, coord: ToroidalCoordinate) -> np.ndarray:
        """Embed toroidal coordinate in E₈ lattice space"""
        
        # Convert to Cartesian
        x, y, z = coord.to_cartesian(self.minor_radius)
        
        # Create 8D embedding using sacred geometry principles
        if coord.digital_root == 9:  # Inward rotational
            # Use convergent spiral pattern in E₈
            embedding = np.array([
                x * self.e8_embedding_scale,
                y * self.e8_embedding_scale,
                z * self.e8_embedding_scale,
                coord.R * math.cos(coord.theta * 9) * self.e8_embedding_scale,
                coord.R * math.sin(coord.theta * 9) * self.e8_embedding_scale,
                coord.sacred_frequency / 1000.0,
                coord.calculate_rotational_energy(),
                coord.digital_root / 9.0
            ])
            
        elif coord.digital_root == 6:  # Outward rotational
            # Use divergent hexagonal pattern in E₈
            embedding = np.array([
                x * self.e8_embedding_scale,
                y * self.e8_embedding_scale,
                z * self.e8_embedding_scale,
                coord.R * math.cos(coord.phi * 6) * self.e8_embedding_scale,
                coord.R * math.sin(coord.phi * 6) * self.e8_embedding_scale,
                coord.sacred_frequency / 1000.0 * self.golden_ratio,
                coord.calculate_rotational_energy() * self.golden_ratio,
                coord.digital_root / 9.0
            ])
            
        elif coord.digital_root == 3:  # Creative seed
            # Use trinity-based pattern in E₈
            embedding = np.array([
                x * self.e8_embedding_scale,
                y * self.e8_embedding_scale,
                z * self.e8_embedding_scale,
                coord.R * math.cos(coord.theta * 3) * self.e8_embedding_scale,
                coord.R * math.sin(coord.phi * 3) * self.e8_embedding_scale,
                coord.sacred_frequency / 1000.0 * self.silver_ratio,
                coord.calculate_rotational_energy() * self.silver_ratio,
                coord.digital_root / 9.0
            ])
            
        else:  # Transformative cycle
            # Use dynamic pattern in E₈
            embedding = np.array([
                x * self.e8_embedding_scale,
                y * self.e8_embedding_scale,
                z * self.e8_embedding_scale,
                coord.R * math.cos(coord.theta * coord.digital_root) * self.e8_embedding_scale,
                coord.R * math.sin(coord.phi * coord.digital_root) * self.e8_embedding_scale,
                coord.sacred_frequency / 1000.0,
                coord.calculate_rotational_energy(),
                coord.digital_root / 9.0
            ])
        
        # Normalize to E₈ lattice scale
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

class ToroidalForceField:
    """Toroidal force field analysis using sacred geometry"""
    
    def __init__(self, geometry: ToroidalSacredGeometry):
        self.geometry = geometry
        self.force_constants = {
            ForceType.GRAVITATIONAL: 6.674e-11,      # G
            ForceType.ELECTROMAGNETIC: 8.854e-12,    # ε₀
            ForceType.NUCLEAR_STRONG: 1.0,           # Normalized
            ForceType.NUCLEAR_WEAK: 1.166e-5         # GF
        }
    
    def calculate_force_vector(self, coord: ToroidalCoordinate, 
                             target_coord: ToroidalCoordinate) -> np.ndarray:
        """Calculate force vector between two toroidal coordinates"""
        
        # Convert to Cartesian
        pos1 = np.array(coord.to_cartesian(self.geometry.minor_radius))
        pos2 = np.array(target_coord.to_cartesian(self.geometry.minor_radius))
        
        # Distance vector
        r_vec = pos2 - pos1
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag == 0:
            return np.zeros(3)
        
        r_hat = r_vec / r_mag
        
        # Force magnitude based on sacred geometry and force type
        force_constant = self.force_constants[coord.force_classification]
        
        # Sacred geometry modulation
        if coord.digital_root == 9:  # Inward/attractive
            force_magnitude = force_constant / (r_mag**2) * coord.calculate_rotational_energy()
            force_vector = -force_magnitude * r_hat  # Attractive
            
        elif coord.digital_root == 6:  # Outward/repulsive
            force_magnitude = force_constant / (r_mag**2) * coord.calculate_rotational_energy()
            force_vector = force_magnitude * r_hat  # Repulsive
            
        elif coord.digital_root == 3:  # Creative/binding
            # Short-range binding force
            force_magnitude = force_constant * math.exp(-r_mag) * coord.calculate_rotational_energy()
            force_vector = -force_magnitude * r_hat  # Binding
            
        else:  # Transformative/decay
            # Weak interaction with angular dependence
            angular_factor = math.cos(coord.theta - target_coord.theta)
            force_magnitude = force_constant * angular_factor * coord.calculate_rotational_energy()
            force_vector = force_magnitude * r_hat
        
        return force_vector
    
    def calculate_toroidal_field_energy(self, shell_points: List[ToroidalCoordinate]) -> float:
        """Calculate total field energy of toroidal shell"""
        
        total_energy = 0.0
        
        for i, coord in enumerate(shell_points):
            coord_energy = 0.0
            
            # Calculate interaction with nearby points
            for j, other_coord in enumerate(shell_points):
                if i != j:
                    force_vector = self.calculate_force_vector(coord, other_coord)
                    force_magnitude = np.linalg.norm(force_vector)
                    
                    # Distance for potential energy
                    pos1 = np.array(coord.to_cartesian(self.geometry.minor_radius))
                    pos2 = np.array(other_coord.to_cartesian(self.geometry.minor_radius))
                    distance = np.linalg.norm(pos2 - pos1)
                    
                    if distance > 0:
                        coord_energy += force_magnitude * distance / 2.0  # Avoid double counting
            
            total_energy += coord_energy
        
        return total_energy
    
    def find_resonant_frequencies(self, shell_points: List[ToroidalCoordinate]) -> Dict[float, List[ToroidalCoordinate]]:
        """Find resonant frequency clusters in toroidal shell"""
        
        frequency_clusters = {}
        
        for coord in shell_points:
            freq = coord.sacred_frequency
            
            if freq not in frequency_clusters:
                frequency_clusters[freq] = []
            
            frequency_clusters[freq].append(coord)
        
        # Analyze resonance patterns
        resonance_analysis = {}
        
        for freq, coords in frequency_clusters.items():
            if len(coords) > 1:  # Resonance requires multiple points
                # Calculate average position and energy
                avg_energy = np.mean([coord.calculate_rotational_energy() for coord in coords])
                
                # Calculate spatial distribution
                positions = [coord.to_cartesian(self.geometry.minor_radius) for coord in coords]
                center = np.mean(positions, axis=0)
                spread = np.std(positions, axis=0)
                
                resonance_analysis[freq] = {
                    'count': len(coords),
                    'average_energy': avg_energy,
                    'spatial_center': center,
                    'spatial_spread': spread,
                    'coordinates': coords
                }
        
        return resonance_analysis

class ToroidalVisualization:
    """Visualization tools for toroidal sacred geometry"""
    
    def __init__(self, geometry: ToroidalSacredGeometry):
        self.geometry = geometry
    
    def plot_toroidal_shell(self, shell_points: List[ToroidalCoordinate], 
                           color_by: str = 'pattern') -> plt.Figure:
        """Plot 3D toroidal shell colored by sacred geometry properties"""
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions and properties
        positions = [coord.to_cartesian(self.geometry.minor_radius) for coord in shell_points]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]
        
        # Color mapping
        if color_by == 'pattern':
            colors = []
            color_map = {
                'INWARD_ROTATIONAL': 'red',
                'OUTWARD_ROTATIONAL': 'blue',
                'CREATIVE_SEED': 'green',
                'TRANSFORMATIVE_CYCLE': 'orange'
            }
            colors = [color_map.get(coord.rotational_pattern, 'gray') for coord in shell_points]
            
        elif color_by == 'force':
            color_map = {
                ForceType.GRAVITATIONAL: 'purple',
                ForceType.ELECTROMAGNETIC: 'yellow',
                ForceType.NUCLEAR_STRONG: 'red',
                ForceType.NUCLEAR_WEAK: 'cyan'
            }
            colors = [color_map.get(coord.force_classification, 'gray') for coord in shell_points]
            
        elif color_by == 'frequency':
            frequencies = [coord.sacred_frequency for coord in shell_points]
            colors = frequencies
            
        else:  # energy
            colors = [coord.calculate_rotational_energy() for coord in shell_points]
        
        # Create scatter plot
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=20, alpha=0.7)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Toroidal Sacred Geometry Shell (colored by {color_by})')
        
        # Add colorbar if numeric
        if color_by in ['frequency', 'energy']:
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        return fig
    
    def plot_force_field_vectors(self, shell_points: List[ToroidalCoordinate], 
                                force_field: ToroidalForceField,
                                sample_rate: int = 10) -> plt.Figure:
        """Plot force field vectors on toroidal shell"""
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points for vector field
        sampled_points = shell_points[::sample_rate]
        
        for coord in sampled_points:
            pos = coord.to_cartesian(self.geometry.minor_radius)
            
            # Calculate average force from nearby points
            nearby_points = [p for p in shell_points if p != coord][:5]  # Limit for performance
            total_force = np.zeros(3)
            
            for nearby in nearby_points:
                force_vec = force_field.calculate_force_vector(coord, nearby)
                total_force += force_vec
            
            # Normalize for visualization
            if np.linalg.norm(total_force) > 0:
                total_force = total_force / np.linalg.norm(total_force) * 0.5
            
            # Plot vector
            ax.quiver(pos[0], pos[1], pos[2], 
                     total_force[0], total_force[1], total_force[2],
                     color='red', alpha=0.7, arrow_length_ratio=0.1)
        
        # Plot shell points
        positions = [coord.to_cartesian(self.geometry.minor_radius) for coord in shell_points]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]
        
        ax.scatter(x_coords, y_coords, z_coords, c='blue', s=10, alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Toroidal Force Field Vectors')
        
        return fig

def demonstrate_toroidal_sacred_geometry():
    """Comprehensive demonstration of toroidal sacred geometry module"""
    
    print("CQE Toroidal Sacred Geometry Module Demonstration")
    print("=" * 60)
    
    # Initialize geometry
    geometry = ToroidalSacredGeometry(major_radius=3.0, minor_radius=1.0)
    
    print(f"Toroidal Parameters:")
    print(f"  Major Radius (R): {geometry.major_radius} (digital root: {geometry.calculate_digital_root(geometry.major_radius)})")
    print(f"  Minor Radius (r): {geometry.minor_radius} (digital root: {geometry.calculate_digital_root(geometry.minor_radius)})")
    print(f"  Golden Ratio: {geometry.golden_ratio:.6f}")
    
    # Generate toroidal shell
    print(f"\nGenerating Toroidal Shell...")
    shell_points = geometry.generate_toroidal_shell(theta_points=18, phi_points=36)  # Reduced for demo
    print(f"Generated {len(shell_points)} shell points")
    
    # Analyze rotational forces
    print(f"\nAnalyzing Rotational Forces...")
    force_analysis = geometry.analyze_rotational_forces(shell_points)
    
    print(f"Pattern Distribution:")
    for pattern, count in force_analysis['pattern_distribution'].items():
        percentage = (count / force_analysis['total_points']) * 100
        print(f"  {pattern}: {count} points ({percentage:.1f}%)")
    
    print(f"\nForce Distribution:")
    for force, count in force_analysis['force_distribution'].items():
        percentage = (count / force_analysis['total_points']) * 100
        print(f"  {force}: {count} points ({percentage:.1f}%)")
    
    print(f"\nEnergy Statistics:")
    stats = force_analysis['energy_statistics']
    print(f"  Mean Energy: {stats['mean']:.6f}")
    print(f"  Energy Std: {stats['std']:.6f}")
    print(f"  Energy Range: {stats['min']:.6f} to {stats['max']:.6f}")
    
    print(f"\nSacred Frequency Distribution:")
    for freq, positions in force_analysis['sacred_frequency_map'].items():
        print(f"  {freq} Hz: {len(positions)} points")
    
    # E₈ embedding analysis
    print(f"\nE₈ Embedding Analysis...")
    sample_coords = shell_points[:5]  # Sample for demonstration
    
    for i, coord in enumerate(sample_coords):
        e8_embedding = geometry.embed_toroidal_in_e8(coord)
        embedding_norm = np.linalg.norm(e8_embedding)
        
        print(f"  Point {i+1}:")
        print(f"    Toroidal: R={coord.R:.3f}, θ={coord.theta:.3f}, φ={coord.phi:.3f}")
        print(f"    Digital Root: {coord.digital_root} → {coord.rotational_pattern}")
        print(f"    Sacred Frequency: {coord.sacred_frequency} Hz")
        print(f"    Force Type: {coord.force_classification.value}")
        print(f"    E₈ Embedding Norm: {embedding_norm:.6f}")
    
    # Force field analysis
    print(f"\nForce Field Analysis...")
    force_field = ToroidalForceField(geometry)
    
    total_field_energy = force_field.calculate_toroidal_field_energy(shell_points[:50])  # Sample for performance
    print(f"Total Field Energy (sample): {total_field_energy:.6f}")
    
    # Resonant frequency analysis
    resonance_analysis = force_field.find_resonant_frequencies(shell_points)
    
    print(f"\nResonant Frequency Clusters:")
    for freq, data in resonance_analysis.items():
        print(f"  {freq} Hz:")
        print(f"    Points: {data['count']}")
        print(f"    Average Energy: {data['average_energy']:.6f}")
        print(f"    Spatial Center: ({data['spatial_center'][0]:.3f}, {data['spatial_center'][1]:.3f}, {data['spatial_center'][2]:.3f})")
    
    # Sacred geometry validation
    print(f"\nSacred Geometry Validation:")
    
    # Test 3-6-9 pattern distribution
    pattern_counts = force_analysis['pattern_distribution']
    total_369_points = (pattern_counts.get('INWARD_ROTATIONAL', 0) + 
                       pattern_counts.get('OUTWARD_ROTATIONAL', 0) + 
                       pattern_counts.get('CREATIVE_SEED', 0))
    
    total_points = force_analysis['total_points']
    sacred_percentage = (total_369_points / total_points) * 100
    
    print(f"  3-6-9 Pattern Coverage: {total_369_points}/{total_points} points ({sacred_percentage:.1f}%)")
    
    # Test golden ratio relationships
    golden_ratio_test = abs(geometry.golden_ratio - 1.618033988749895) < 1e-10
    print(f"  Golden Ratio Precision: {golden_ratio_test}")
    
    # Test sacred frequency alignment
    expected_frequencies = {432.0, 528.0, 396.0, 741.0}
    found_frequencies = set(force_analysis['sacred_frequency_map'].keys())
    frequency_alignment = expected_frequencies.issubset(found_frequencies)
    print(f"  Sacred Frequency Alignment: {frequency_alignment}")
    
    print(f"\nToroidal Sacred Geometry Module Demonstration Complete!")
    
    return {
        'geometry': geometry,
        'shell_points': shell_points,
        'force_analysis': force_analysis,
        'force_field': force_field,
        'resonance_analysis': resonance_analysis
    }

if __name__ == "__main__":
    # Run comprehensive demonstration
    demo_results = demonstrate_toroidal_sacred_geometry()
    
    # Optional: Create visualizations (requires matplotlib)
    try:
        print(f"\nCreating Visualizations...")
        
        geometry = demo_results['geometry']
        shell_points = demo_results['shell_points']
        force_field = demo_results['force_field']
        
        # Create visualization object
        viz = ToroidalVisualization(geometry)
        
        # Plot shell colored by pattern
        fig1 = viz.plot_toroidal_shell(shell_points, color_by='pattern')
        fig1.savefig(str(Path(__file__).parent / 'toroidal_shell_patterns.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: toroidal_shell_patterns.png")
        
        # Plot shell colored by force type
        fig2 = viz.plot_toroidal_shell(shell_points, color_by='force')
        fig2.savefig(str(Path(__file__).parent / 'toroidal_shell_forces.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: toroidal_shell_forces.png")
        
        # Plot force field vectors
        fig3 = viz.plot_force_field_vectors(shell_points, force_field, sample_rate=20)
        fig3.savefig(str(Path(__file__).parent / 'toroidal_force_vectors.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: toroidal_force_vectors.png")
        
        plt.close('all')  # Clean up
        
    except ImportError:
        print(f"  Matplotlib not available for visualizations")
    except Exception as e:
        print(f"  Visualization error: {e}")
    
    print(f"\nModule demonstration complete with {len(demo_results['shell_points'])} toroidal points analyzed.")
