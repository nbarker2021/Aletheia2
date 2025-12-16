"""
Quaternion Operations for CQE

Quaternions provide a natural representation for rotations in 3D and 4D space.
In the CQE framework, quaternions are used for:
- E8 rotations (via pairs of quaternions - octonions)
- Weyl group operations
- Toroidal flow rotations
- Morphonic transformations

Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class Quaternion:
    """
    Quaternion representation: q = w + xi + yj + zk
    
    Where i² = j² = k² = ijk = -1
    """
    w: float  # Real part
    x: float  # i component
    y: float  # j component
    z: float  # k component
    
    def __post_init__(self):
        """Ensure components are floats."""
        self.w = float(self.w)
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """Create quaternion from array [w, x, y, z]."""
        if len(arr) != 4:
            raise ValueError(f"Expected 4 components, got {len(arr)}")
        return cls(arr[0], arr[1], arr[2], arr[3])
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        """Return identity quaternion (1, 0, 0, 0)."""
        return cls(1, 0, 0, 0)
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """
        Create quaternion from axis-angle representation.
        
        Args:
            axis: 3D unit vector
            angle: Rotation angle in radians
        
        Returns:
            Quaternion representing the rotation
        """
        axis = axis / np.linalg.norm(axis)  # Normalize
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        
        return cls(
            w=np.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])
    
    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Convert to axis-angle representation.
        
        Returns:
            (axis, angle) tuple
        """
        angle = 2 * np.arccos(np.clip(self.w, -1, 1))
        
        if abs(angle) < 1e-10:
            # No rotation
            return np.array([1, 0, 0]), 0.0
        
        sin_half = np.sin(angle / 2)
        if abs(sin_half) < 1e-10:
            return np.array([1, 0, 0]), 0.0
        
        axis = np.array([self.x, self.y, self.z]) / sin_half
        return axis, angle
    
    def norm(self) -> float:
        """Compute quaternion norm."""
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Quaternion':
        """Return normalized quaternion (unit quaternion)."""
        n = self.norm()
        if n < 1e-10:
            return Quaternion.identity()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def conjugate(self) -> 'Quaternion':
        """Return quaternion conjugate."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        """Return quaternion inverse."""
        n_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n_sq < 1e-10:
            raise ValueError("Cannot invert zero quaternion")
        
        conj = self.conjugate()
        return Quaternion(conj.w/n_sq, conj.x/n_sq, conj.y/n_sq, conj.z/n_sq)
    
    def __mul__(self, other: Union['Quaternion', float]) -> 'Quaternion':
        """Quaternion multiplication."""
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, 
                            self.y * other, self.z * other)
        
        if not isinstance(other, Quaternion):
            raise TypeError(f"Cannot multiply Quaternion with {type(other)}")
        
        # Hamilton product
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        
        return Quaternion(w, x, y, z)
    
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion addition."""
        if not isinstance(other, Quaternion):
            raise TypeError(f"Cannot add Quaternion with {type(other)}")
        
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion subtraction."""
        if not isinstance(other, Quaternion):
            raise TypeError(f"Cannot subtract {type(other)} from Quaternion")
        
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector using this quaternion.
        
        Uses the formula: v' = q * v * q^(-1)
        where v is treated as a pure quaternion (0, v)
        
        Args:
            v: 3D vector to rotate
        
        Returns:
            Rotated 3D vector
        """
        if len(v) != 3:
            raise ValueError(f"Expected 3D vector, got {len(v)}D")
        
        # Convert vector to pure quaternion
        v_quat = Quaternion(0, v[0], v[1], v[2])
        
        # Rotate: q * v * q^(-1)
        rotated = self * v_quat * self.conjugate()
        
        return np.array([rotated.x, rotated.y, rotated.z])
    
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to 3×3 rotation matrix.
        
        Returns:
            3×3 rotation matrix
        """
        # Normalize first
        q = self.normalize()
        
        w, x, y, z = q.w, q.x, q.y, q.z
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> 'Quaternion':
        """
        Create quaternion from 3×3 rotation matrix.
        
        Args:
            R: 3×3 rotation matrix
        
        Returns:
            Quaternion
        """
        if R.shape != (3, 3):
            raise ValueError(f"Expected 3×3 matrix, got {R.shape}")
        
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return cls(w, x, y, z)
    
    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """
        Spherical linear interpolation between two quaternions.
        
        Args:
            other: Target quaternion
            t: Interpolation parameter [0, 1]
        
        Returns:
            Interpolated quaternion
        """
        # Normalize both quaternions
        q1 = self.normalize()
        q2 = other.normalize()
        
        # Compute dot product
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        
        # If dot < 0, negate q2 to take shorter path
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        # Clamp dot to avoid numerical issues
        dot = np.clip(dot, -1, 1)
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z)
            )
            return result.normalize()
        
        # Compute angle and interpolate
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return Quaternion(
            q1.w * w1 + q2.w * w2,
            q1.x * w1 + q2.x * w2,
            q1.y * w1 + q2.y * w2,
            q1.z * w1 + q2.z * w2
        )
    
    def __repr__(self) -> str:
        return f"Quaternion({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
    
    def __str__(self) -> str:
        return f"{self.w:.4f} + {self.x:.4f}i + {self.y:.4f}j + {self.z:.4f}k"
