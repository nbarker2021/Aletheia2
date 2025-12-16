"""
CQE-GVS: Complete Generative Video System
Real-time, lossless video generation via E8 geometric projection
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import time

from .core.e8_ops import E8Lattice, ALENAOps, generate_e8_state
from .core.toroidal_geometry import ToroidalFlow, DihedralSymmetry
from .worlds.world_forge import WorldForge, WorldManifold, WorldType
from .rendering.render_engine import GeometricRenderer, RenderConfig, WeylChamberStyler


@dataclass
class VideoSpec:
    """Video generation specification."""
    prompt: str
    duration: float  # seconds
    fps: float = 30.0
    resolution: Tuple[int, int] = (1920, 1080)
    world_type: WorldType = WorldType.NATURAL
    seed: Optional[int] = None
    
    def total_frames(self) -> int:
        return int(self.duration * self.fps)


class CQEGenerativeVideoSystem:
    """
    Complete CQE Generative Video System.
    
    Generates video via:
    1. Text prompt → E8 state (encoding)
    2. E8 state → World manifold (WorldForge)
    3. World → Trajectory (toroidal flow)
    4. Trajectory → Frames (rendering)
    5. Frames → Video file (output)
    """
    
    def __init__(self, coupling: float = 0.03):
        self.coupling = coupling
        
        # Core components
        self.e8 = E8Lattice()
        self.alena = ALENAOps(self.e8)
        self.flow = ToroidalFlow(coupling=coupling)
        self.dihedral = DihedralSymmetry(order=24)
        
        # High-level components
        self.forge = WorldForge()
        self.renderer = None  # Created per-video based on spec
        self.styler = WeylChamberStyler()
        
        print("✓ CQE-GVS initialized")
        print(f"  Coupling: {self.coupling}")
        print(f"  E8 roots: {len(self.e8.roots)}")
        print(f"  Weyl chambers: {len(self.e8.weyl_chambers)}")
    
    def encode_prompt(self, prompt: str, seed: Optional[int] = None) -> np.ndarray:
        """
        Encode text prompt to E8 state.
        
        Uses digital root mapping and semantic analysis.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Compute digital root from prompt
        total = sum(ord(c) for c in prompt)
        while total >= 10:
            total = sum(int(d) for d in str(total))
        dr = total if total > 0 else 9
        
        print(f"  Prompt DR: {dr}")
        
        # Generate E8 state biased by digital root
        e8_state = np.random.randn(8)
        
        # Emphasize dimension corresponding to DR
        e8_state[dr % 8] *= 2.0
        
        # Add semantic weighting based on keywords
        keywords = {
            'fast': [0, 1],      # EM dimensions
            'slow': [2, 3],      # Weak dimensions
            'strong': [4, 5],    # Strong dimensions
            'gentle': [6, 7],    # Gravity dimensions
            'bright': [0, 4],
            'dark': [2, 6],
            'colorful': [4, 5, 6],
            'simple': [0, 1, 2],
            'complex': [5, 6, 7]
        }
        
        prompt_lower = prompt.lower()
        for keyword, dims in keywords.items():
            if keyword in prompt_lower:
                for dim in dims:
                    e8_state[dim] *= 1.5
        
        # Normalize to E8 manifold
        norm = np.linalg.norm(e8_state)
        if norm > 0:
            e8_state = e8_state / norm * np.sqrt(2)
        
        return e8_state
    
    def generate_video(self, spec: VideoSpec, output_path: str,
                      verbose: bool = True) -> dict:
        """
        Generate complete video from specification.
        
        Args:
            spec: Video specification
            output_path: Output video file path
            verbose: Print progress
            
        Returns:
            dict with generation statistics
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CQE-GVS Video Generation")
            print(f"{'='*60}")
            print(f"Prompt: \"{spec.prompt}\"")
            print(f"Duration: {spec.duration}s @ {spec.fps} FPS")
            print(f"Resolution: {spec.resolution[0]}x{spec.resolution[1]}")
            print(f"World: {spec.world_type.value}")
            print(f"Total frames: {spec.total_frames()}")
            print()
        
        # Step 1: Encode prompt to E8
        if verbose:
            print("Step 1: Encoding prompt to E8 space...")
        
        e8_state = self.encode_prompt(spec.prompt, spec.seed)
        weyl_chamber = self.e8.find_weyl_chamber(e8_state)
        digital_root = self.e8.compute_digital_root(e8_state)
        
        if verbose:
            print(f"  E8 state: {e8_state}")
            print(f"  Weyl chamber: {weyl_chamber}/48")
            print(f"  Digital root: {digital_root}")
            print()
        
        # Step 2: Spawn world manifold
        if verbose:
            print("Step 2: Spawning world manifold...")
        
        manifold = self.forge.spawn(
            spec.world_type,
            hypothesis=spec.prompt,
            seed=spec.seed
        )
        
        if verbose:
            print(f"  World type: {manifold.world_type.value}")
            print(f"  Complexity: {manifold.complexity:.2f}")
            print(f"  Curvature: {manifold.curvature:.2f}")
            print(f"  Objects: {len(manifold.objects)}")
            print()
        
        # Step 3: Generate trajectory
        if verbose:
            print("Step 3: Generating temporal trajectory...")
        
        trajectory = self.forge.evolve_world(
            manifold,
            duration=spec.duration,
            fps=spec.fps
        )
        
        is_closed = self.flow.check_closure(trajectory)
        
        if verbose:
            print(f"  Frames: {len(trajectory)}")
            print(f"  Closed loop: {is_closed}")
            print()
        
        # Step 4: Render frames
        if verbose:
            print("Step 4: Rendering frames...")
        
        # Create renderer with spec resolution
        render_config = RenderConfig(
            resolution=spec.resolution,
            fps=spec.fps
        )
        self.renderer = GeometricRenderer(render_config)
        
        frames = self.renderer.render_trajectory(
            trajectory,
            manifold=manifold,
            fast=True
        )
        
        if verbose:
            print(f"  Rendered: {len(frames)} frames")
            print()
        
        # Step 5: Apply Weyl chamber styling
        if verbose:
            print("Step 5: Applying Weyl chamber styling...")
        
        styled_frames = []
        for frame in frames:
            styled = self.styler.apply_style(frame, weyl_chamber)
            styled_frames.append(styled)
        
        if verbose:
            print(f"  Style: {self.styler.get_style(weyl_chamber)}")
            print()
        
        # Step 6: Save video
        if verbose:
            print("Step 6: Saving video...")
        
        self.renderer.save_video(styled_frames, output_path, spec.fps)
        
        # Compute statistics
        end_time = time.time()
        elapsed = end_time - start_time
        fps_actual = len(frames) / elapsed
        
        stats = {
            'frames': len(frames),
            'duration': spec.duration,
            'fps_target': spec.fps,
            'fps_actual': fps_actual,
            'elapsed_time': elapsed,
            'weyl_chamber': weyl_chamber,
            'digital_root': digital_root,
            'world_type': spec.world_type.value,
            'is_closed': is_closed,
            'output_path': output_path
        }
        
        if verbose:
            print()
            print(f"{'='*60}")
            print(f"Generation Complete")
            print(f"{'='*60}")
            print(f"Elapsed time: {elapsed:.2f}s")
            print(f"Rendering speed: {fps_actual:.1f} FPS")
            print(f"Real-time factor: {fps_actual / spec.fps:.2f}x")
            print(f"Output: {output_path}")
            print()
        
        return stats
    
    def generate_with_keyframes(self, spec: VideoSpec,
                               keyframes: List[Tuple[float, str]],
                               output_path: str) -> dict:
        """
        Generate video with keyframe control.
        
        Args:
            spec: Base video specification
            keyframes: List of (time, prompt) keyframes
            output_path: Output video file path
            
        Returns:
            Generation statistics
        """
        print(f"\nGenerating video with {len(keyframes)} keyframes...")
        
        # Encode all keyframes to E8
        keyframe_states = []
        for time, prompt in keyframes:
            e8_state = self.encode_prompt(prompt, spec.seed)
            keyframe_states.append((time, e8_state))
        
        # Generate trajectory segments
        all_frames = []
        
        for i in range(len(keyframe_states) - 1):
            t_start, state_start = keyframe_states[i]
            t_end, state_end = keyframe_states[i + 1]
            
            segment_duration = t_end - t_start
            segment_frames = int(segment_duration * spec.fps)
            
            print(f"  Segment {i+1}: {t_start:.1f}s → {t_end:.1f}s ({segment_frames} frames)")
            
            # Interpolate between keyframes
            segment_trajectory = []
            for j in range(segment_frames):
                t = j / (segment_frames - 1) if segment_frames > 1 else 0
                state = self.e8.interpolate_geodesic(state_start, state_end, t)
                segment_trajectory.append(state)
            
            # Render segment
            render_config = RenderConfig(resolution=spec.resolution, fps=spec.fps)
            self.renderer = GeometricRenderer(render_config)
            
            segment_frames = self.renderer.render_trajectory(segment_trajectory, fast=True)
            all_frames.extend(segment_frames)
        
        # Save video
        self.renderer.save_video(all_frames, output_path, spec.fps)
        
        return {
            'frames': len(all_frames),
            'keyframes': len(keyframes),
            'output_path': output_path
        }
    
    def morph_worlds(self, world1_prompt: str, world2_prompt: str,
                    duration: float, output_path: str,
                    world1_type: WorldType = WorldType.NATURAL,
                    world2_type: WorldType = WorldType.COSMIC,
                    resolution: Tuple[int, int] = (1920, 1080),
                    fps: float = 30.0) -> dict:
        """
        Generate video morphing between two worlds.
        
        Args:
            world1_prompt: First world prompt
            world2_prompt: Second world prompt
            duration: Morph duration in seconds
            output_path: Output video file path
            world1_type: First world type
            world2_type: Second world type
            resolution: Video resolution
            fps: Frames per second
            
        Returns:
            Generation statistics
        """
        print(f"\nMorphing worlds:")
        print(f"  {world1_type.value}: \"{world1_prompt}\"")
        print(f"  → {world2_type.value}: \"{world2_prompt}\"")
        print()
        
        # Spawn both worlds
        world1 = self.forge.spawn(world1_type, world1_prompt, seed=1)
        world2 = self.forge.spawn(world2_type, world2_prompt, seed=2)
        
        # Generate morph trajectory
        num_frames = int(duration * fps)
        trajectory = self.forge.interpolate_worlds(world1, world2, num_frames)
        
        print(f"Morph trajectory: {len(trajectory)} frames\n")
        
        # Render
        render_config = RenderConfig(resolution=resolution, fps=fps)
        self.renderer = GeometricRenderer(render_config)
        
        # Interpolate manifold properties for rendering
        frames = []
        for i, e8_state in enumerate(trajectory):
            t = i / (num_frames - 1) if num_frames > 1 else 0
            
            # Create interpolated manifold
            # (simplified - just use world1 properties)
            frame = self.renderer.render_frame_fast(e8_state, world1)
            frames.append(frame)
        
        # Save
        self.renderer.save_video(frames, output_path, fps)
        
        return {
            'frames': len(frames),
            'world1': world1_type.value,
            'world2': world2_type.value,
            'output_path': output_path
        }


if __name__ == "__main__":
    # Test complete system
    print("=== CQE-GVS Complete System Test ===\n")
    
    # Initialize system
    gvs = CQEGenerativeVideoSystem(coupling=0.03)
    
    # Test 1: Simple video generation
    print("\nTest 1: Simple video generation")
    print("-" * 60)
    
    spec1 = VideoSpec(
        prompt="A serene forest with sunlight filtering through trees",
        duration=3.0,
        fps=30,
        resolution=(640, 480),
        world_type=WorldType.NATURAL,
        seed=42
    )
    
    stats1 = gvs.generate_video(spec1, "test_forest.mp4", verbose=True)
    
    # Test 2: Keyframe video
    print("\nTest 2: Keyframe-controlled video")
    print("-" * 60)
    
    spec2 = VideoSpec(
        prompt="Morphing scene",
        duration=6.0,
        fps=30,
        resolution=(640, 480),
        world_type=WorldType.NATURAL,
        seed=123
    )
    
    keyframes = [
        (0.0, "A peaceful meadow at dawn"),
        (2.0, "The same meadow at noon"),
        (4.0, "The meadow at sunset"),
        (6.0, "The meadow under stars")
    ]
    
    stats2 = gvs.generate_with_keyframes(spec2, keyframes, "test_meadow_day.mp4")
    
    # Test 3: World morphing
    print("\nTest 3: World morphing")
    print("-" * 60)
    
    stats3 = gvs.morph_worlds(
        world1_prompt="A lush green forest",
        world2_prompt="A vast cosmic nebula",
        duration=5.0,
        output_path="test_morph.mp4",
        world1_type=WorldType.NATURAL,
        world2_type=WorldType.COSMIC,
        resolution=(640, 480),
        fps=30
    )
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)

