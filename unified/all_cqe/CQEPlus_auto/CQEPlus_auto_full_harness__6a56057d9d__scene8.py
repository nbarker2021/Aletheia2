#!/usr/bin/env python3
"""
ScenE8 CLI
Command-line interface for ScenE8 Generative Video AI
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import ScenE8Engine, ScenE8Config, GenerationMode


def main():
    parser = argparse.ArgumentParser(
        description="ScenE8: Geometry-First Generative Video AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scene8 generate "A golden spiral unfolds" -o video.mp4
  scene8 generate "Cosmic dance" --duration 10 --fps 30 -o output.mp4
  scene8 generate "Abstract geometry" --resolution 1920x1080 -o hd.mp4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate video from prompt')
    gen_parser.add_argument('prompt', type=str, help='Text prompt for generation')
    gen_parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    gen_parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds (default: 5.0)')
    gen_parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    gen_parser.add_argument('--resolution', type=str, default='1280x720', help='Resolution WxH (default: 1280x720)')
    gen_parser.add_argument('--no-loop', action='store_true', help='Disable toroidal closure (seamless looping)')
    gen_parser.add_argument('--no-symmetry', action='store_true', help='Disable dihedral symmetry')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'version':
        print("ScenE8 v1.0.0-MVP")
        print("Geometry-First Generative Video AI")
        print("Built on CQE Framework")
        return
    
    if args.command == 'generate':
        # Parse resolution
        try:
            width, height = map(int, args.resolution.split('x'))
        except:
            print(f"Error: Invalid resolution format '{args.resolution}'. Use WxH (e.g., 1920x1080)")
            sys.exit(1)
        
        # Create config
        config = ScenE8Config(
            resolution=(width, height),
            fps=args.fps,
            duration=args.duration,
            use_toroidal_closure=not args.no_loop,
            use_dihedral_symmetry=not args.no_symmetry
        )
        
        # Initialize engine
        print(f"🎬 ScenE8: Initializing engine...")
        engine = ScenE8Engine(config)
        
        # Generate
        print(f"🎨 Generating video from prompt: '{args.prompt}'")
        print(f"   Resolution: {width}x{height}")
        print(f"   Duration: {args.duration}s @ {args.fps} fps")
        print(f"   Frames: {int(args.duration * args.fps)}")
        
        result = engine.generate(args.prompt)
        
        # Save
        print(f"💾 Saving to: {args.output}")
        engine.save_video(result, args.output)
        
        print(f"✅ Done! Generated {result['metadata']['num_frames']} frames")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
