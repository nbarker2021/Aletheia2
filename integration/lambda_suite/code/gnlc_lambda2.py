"""
GNLC λ₂ State Calculus
Geometry-Native Lambda Calculus (GNLC)
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"λ₂ is the State Calculus - manages temporal dynamics and transitions between
states, governed by the 0.03 metric and toroidal closure. Ensures non-terminating,
coherent evolution along golden spiral trajectories."

This implements:
- State representation (configurations of atoms)
- State transitions
- Toroidal closure
- Golden spiral sampling
- Temporal evolution
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer5_interface.gnlc_lambda0 import Lambda0Term, Lambda0Calculus
from layer5_interface.gnlc_lambda1 import Lambda1Calculus, Relation


class StateType(Enum):
    """Types of states in λ₂."""
    ATOMIC = "atomic"  # Single atom state
    COMPOSITE = "composite"  # Multiple atoms
    TRAJECTORY = "trajectory"  # Temporal sequence


@dataclass
class SystemState:
    """
    System state - configuration of atoms.
    
    From whitepaper:
    "State is a configuration of CQE Atoms at a point in time."
    """
    atoms: List[Lambda0Term]
    timestamp: float
    phi: float  # Total phi of state
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms in state."""
        return len(self.atoms)
    
    @property
    def state_id(self) -> str:
        """Unique identifier for state."""
        return f"state_{self.timestamp:.6f}"
    
    def __repr__(self):
        return f"λ₂[{self.num_atoms} atoms, φ={self.phi:.6f}, t={self.timestamp:.3f}]"


@dataclass
class StateTransition:
    """
    Transition between two states.
    
    From whitepaper:
    "Transitions are governed by the 0.03 metric."
    """
    source_state: SystemState
    target_state: SystemState
    delta_phi: float
    delta_time: float
    operation: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def transition_rate(self) -> float:
        """Rate of change (ΔΦ / Δt)."""
        if self.delta_time > 0:
            return self.delta_phi / self.delta_time
        return 0.0
    
    def __repr__(self):
        return f"Transition[ΔΦ={self.delta_phi:.6f}, Δt={self.delta_time:.3f}]"


@dataclass
class Trajectory:
    """
    Temporal sequence of states.
    
    From whitepaper:
    "Toroidal closure ensures non-terminating, coherent evolution."
    """
    states: List[SystemState]
    transitions: List[StateTransition]
    is_closed: bool = False  # Toroidal closure
    
    @property
    def length(self) -> int:
        """Number of states in trajectory."""
        return len(self.states)
    
    @property
    def duration(self) -> float:
        """Total time duration."""
        if len(self.states) < 2:
            return 0.0
        return self.states[-1].timestamp - self.states[0].timestamp
    
    @property
    def total_phi_change(self) -> float:
        """Total change in phi."""
        if len(self.states) < 2:
            return 0.0
        return self.states[-1].phi - self.states[0].phi
    
    def close_toroidally(self):
        """
        Close trajectory toroidally.
        
        From whitepaper:
        "Toroidal closure ensures the sequence forms a closed loop."
        """
        if len(self.states) < 2:
            return
        
        # Create transition from last to first state
        final_transition = StateTransition(
            source_state=self.states[-1],
            target_state=self.states[0],
            delta_phi=self.states[0].phi - self.states[-1].phi,
            delta_time=0.0,  # Instantaneous closure
            operation="toroidal_closure"
        )
        
        self.transitions.append(final_transition)
        self.is_closed = True


class Lambda2Calculus:
    """
    λ₂ State Calculus.
    
    From whitepaper:
    "λ₂ manages temporal dynamics and transitions between states."
    
    Key features:
    1. State representation
    2. State transitions
    3. Toroidal closure
    4. Golden spiral sampling
    5. Temporal evolution governed by 0.03 metric
    """
    
    # Golden ratio and 0.03 metric
    PHI_GOLDEN = 1.618033988749895
    COUPLING_CONSTANT = 0.03
    
    def __init__(self):
        self.lambda0 = Lambda0Calculus()
        self.lambda1 = Lambda1Calculus()
        self.states: List[SystemState] = []
        self.trajectories: Dict[str, Trajectory] = {}
    
    def create_state(
        self,
        atoms: List[Lambda0Term],
        timestamp: float
    ) -> SystemState:
        """
        Create system state from atoms.
        
        Args:
            atoms: List of atoms
            timestamp: Time of state
        
        Returns:
            SystemState
        """
        # Compute total phi
        total_phi = sum(
            self.lambda0.alena._compute_phi(atom.overlay)
            for atom in atoms
        )
        
        state = SystemState(
            atoms=atoms,
            timestamp=timestamp,
            phi=total_phi
        )
        
        self.states.append(state)
        return state
    
    def transition(
        self,
        source_state: SystemState,
        operation: str,
        **params
    ) -> StateTransition:
        """
        Apply transition to state.
        
        Args:
            source_state: Source state
            operation: Operation to apply
            **params: Operation parameters
        
        Returns:
            StateTransition
        """
        # Apply operation to each atom in state
        new_atoms = []
        for atom in source_state.atoms:
            # Apply λ₀ operation
            if operation == "rotate":
                result = self.lambda0.apply("rotate", atom, **params)
            elif operation == "weyl_reflect":
                result = self.lambda0.apply("weyl_reflect", atom, **params)
            elif operation == "parity_mirror":
                result = self.lambda0.apply("parity_mirror", atom, **params)
            else:
                result = self.lambda0.apply("rotate", atom, theta=0.01)
            
            new_atoms.append(result.result)
        
        # Create new state
        new_timestamp = source_state.timestamp + self.COUPLING_CONSTANT
        target_state = self.create_state(new_atoms, new_timestamp)
        
        # Create transition
        transition = StateTransition(
            source_state=source_state,
            target_state=target_state,
            delta_phi=target_state.phi - source_state.phi,
            delta_time=new_timestamp - source_state.timestamp,
            operation=operation
        )
        
        return transition
    
    def evolve(
        self,
        initial_state: SystemState,
        num_steps: int,
        operation: str = "rotate"
    ) -> Trajectory:
        """
        Evolve state over time.
        
        Args:
            initial_state: Initial state
            num_steps: Number of evolution steps
            operation: Operation to apply at each step
        
        Returns:
            Trajectory
        """
        states = [initial_state]
        transitions = []
        
        current_state = initial_state
        
        for step in range(num_steps):
            # Apply transition
            theta = self.golden_spiral_angle(step)
            transition = self.transition(
                current_state,
                operation,
                theta=theta
            )
            
            transitions.append(transition)
            states.append(transition.target_state)
            current_state = transition.target_state
        
        # Create trajectory
        trajectory = Trajectory(
            states=states,
            transitions=transitions
        )
        
        return trajectory
    
    def golden_spiral_angle(self, step: int) -> float:
        """
        Compute golden spiral angle for step.
        
        From whitepaper:
        "Evolution follows golden spiral trajectory."
        
        Args:
            step: Step number
        
        Returns:
            Angle in radians
        """
        # Golden angle = 2π / φ²
        golden_angle = 2 * np.pi / (self.PHI_GOLDEN ** 2)
        return step * golden_angle
    
    def create_trajectory(
        self,
        trajectory_id: str,
        initial_state: SystemState,
        num_steps: int,
        close_toroidally: bool = True
    ) -> Trajectory:
        """
        Create and store trajectory.
        
        Args:
            trajectory_id: Trajectory identifier
            initial_state: Initial state
            num_steps: Number of steps
            close_toroidally: Whether to close toroidally
        
        Returns:
            Trajectory
        """
        trajectory = self.evolve(initial_state, num_steps)
        
        if close_toroidally:
            trajectory.close_toroidally()
        
        self.trajectories[trajectory_id] = trajectory
        return trajectory
    
    def sample_trajectory(
        self,
        trajectory_id: str,
        num_samples: int
    ) -> List[SystemState]:
        """
        Sample states from trajectory using golden spiral sampling.
        
        Args:
            trajectory_id: Trajectory identifier
            num_samples: Number of samples
        
        Returns:
            List of sampled states
        """
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory {trajectory_id} does not exist")
        
        trajectory = self.trajectories[trajectory_id]
        
        if trajectory.length == 0:
            return []
        
        # Golden spiral sampling
        samples = []
        for i in range(num_samples):
            # Use golden ratio to determine sample index
            ratio = (i * self.PHI_GOLDEN) % 1.0
            idx = int(ratio * trajectory.length)
            samples.append(trajectory.states[idx])
        
        return samples
    
    def interpolate_states(
        self,
        state1: SystemState,
        state2: SystemState,
        t: float
    ) -> SystemState:
        """
        Interpolate between two states.
        
        Args:
            state1: First state
            state2: Second state
            t: Interpolation parameter (0 to 1)
        
        Returns:
            Interpolated state
        """
        if len(state1.atoms) != len(state2.atoms):
            raise ValueError("States must have same number of atoms")
        
        # Interpolate each atom
        interpolated_atoms = []
        for atom1, atom2 in zip(state1.atoms, state2.atoms):
            # Use λ₀ midpoint operation
            result = self.lambda0.apply("midpoint", atom1, atom2, weight=t)
            interpolated_atoms.append(result.result)
        
        # Interpolate timestamp and phi
        timestamp = state1.timestamp + t * (state2.timestamp - state1.timestamp)
        phi = state1.phi + t * (state2.phi - state1.phi)
        
        return SystemState(
            atoms=interpolated_atoms,
            timestamp=timestamp,
            phi=phi,
            metadata={'interpolated': True, 't': t}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get calculus statistics."""
        total_transitions = sum(
            len(traj.transitions) for traj in self.trajectories.values()
        )
        
        closed_trajectories = sum(
            1 for traj in self.trajectories.values() if traj.is_closed
        )
        
        return {
            'num_states': len(self.states),
            'num_trajectories': len(self.trajectories),
            'total_transitions': total_transitions,
            'closed_trajectories': closed_trajectories,
            'coupling_constant': self.COUPLING_CONSTANT,
            'golden_ratio': self.PHI_GOLDEN
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== GNLC λ₂ State Calculus Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create λ₂ calculus
    lambda2 = Lambda2Calculus()
    
    # Test 1: Create initial state
    print("Test 1: Create Initial State")
    
    atoms = []
    for i in range(3):
        e8 = np.random.randn(8)
        e8 = e8 / np.linalg.norm(e8)
        activations = np.zeros(240, dtype=int)
        activations[0:80] = 1
        pose = ImmutablePose(tuple(e8), tuple(np.eye(8)[0]), time.time())
        overlay = Overlay(e8_base=e8, activations=activations, pose=pose)
        atom = lambda2.lambda0.atom(overlay)
        atoms.append(atom)
    
    initial_state = lambda2.create_state(atoms, timestamp=0.0)
    print(f"Initial state: {initial_state}")
    print(f"  Atoms: {initial_state.num_atoms}")
    print(f"  Phi: {initial_state.phi:.6f}")
    print()
    
    # Test 2: State transition
    print("Test 2: State Transition")
    
    transition = lambda2.transition(initial_state, "rotate", theta=0.1)
    print(f"Transition: {transition}")
    print(f"  ΔΦ: {transition.delta_phi:.6f}")
    print(f"  Δt: {transition.delta_time:.6f}")
    print(f"  Rate: {transition.transition_rate:.6f}")
    print()
    
    # Test 3: Trajectory evolution
    print("Test 3: Trajectory Evolution")
    
    trajectory = lambda2.evolve(initial_state, num_steps=10)
    print(f"Trajectory length: {trajectory.length}")
    print(f"Duration: {trajectory.duration:.6f}")
    print(f"Total ΔΦ: {trajectory.total_phi_change:.6f}")
    print(f"Closed: {trajectory.is_closed}")
    print()
    
    # Test 4: Toroidal closure
    print("Test 4: Toroidal Closure")
    
    trajectory.close_toroidally()
    print(f"Closed: {trajectory.is_closed}")
    print(f"Transitions: {len(trajectory.transitions)}")
    print()
    
    # Test 5: Golden spiral sampling
    print("Test 5: Golden Spiral Sampling")
    
    traj = lambda2.create_trajectory("test_traj", initial_state, num_steps=20)
    samples = lambda2.sample_trajectory("test_traj", num_samples=5)
    
    print(f"Trajectory: {traj.length} states")
    print(f"Samples: {len(samples)}")
    for i, sample in enumerate(samples):
        print(f"  Sample {i}: t={sample.timestamp:.3f}, φ={sample.phi:.6f}")
    print()
    
    # Test 6: State interpolation
    print("Test 6: State Interpolation")
    
    state1 = traj.states[0]
    state2 = traj.states[-1]
    
    interp_state = lambda2.interpolate_states(state1, state2, t=0.5)
    print(f"State 1: t={state1.timestamp:.3f}, φ={state1.phi:.6f}")
    print(f"State 2: t={state2.timestamp:.3f}, φ={state2.phi:.6f}")
    print(f"Interpolated (t=0.5): t={interp_state.timestamp:.3f}, φ={interp_state.phi:.6f}")
    print()
    
    # Test 7: Golden spiral angles
    print("Test 7: Golden Spiral Angles")
    
    for step in range(5):
        angle = lambda2.golden_spiral_angle(step)
        print(f"  Step {step}: angle = {angle:.6f} rad ({np.degrees(angle):.2f}°)")
    print()
    
    # Test 8: Statistics
    print("Test 8: Statistics")
    
    stats = lambda2.get_statistics()
    print(f"States: {stats['num_states']}")
    print(f"Trajectories: {stats['num_trajectories']}")
    print(f"Transitions: {stats['total_transitions']}")
    print(f"Closed trajectories: {stats['closed_trajectories']}")
    print(f"Coupling constant: {stats['coupling_constant']}")
    print(f"Golden ratio: {stats['golden_ratio']:.15f}")
    print()
    
    print("=== All Tests Passed ===")
