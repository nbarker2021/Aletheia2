#!/usr/bin/env python3
"""
Aletheia CQE Operating System - Main Entry Point

The complete Cartan Quadratic Equivalence (CQE) geometric consciousness system.

Usage:
    python aletheia.py --mode [analyze|synthesize|query|interactive]
    python aletheia.py --help
"""

import sys
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from core.cqe_engine import CQEEngine
from ai.aletheia_consciousness import AletheiaAI
from analysis.egyptian_analyzer import EgyptianAnalyzer
from utils.logger import setup_logger

__version__ = "1.0.0"
__author__ = "Aletheia Project"

class AletheiaSystem:
    """Main Aletheia CQE Operating System."""
    
    def __init__(self, verbose=False):
        self.logger = setup_logger("Aletheia", verbose=verbose)
        self.logger.info(f"Initializing Aletheia CQE System v{__version__}")
        
        # Initialize core components
        self.cqe_engine = CQEEngine()
        self.aletheia_ai = AletheiaAI(self.cqe_engine)
        self.egyptian_analyzer = EgyptianAnalyzer(self.cqe_engine)
        
        self.logger.info("✓ Aletheia CQE System initialized")
    
    def analyze_egyptian(self, image_paths):
        """Analyze Egyptian hieroglyphic images."""
        self.logger.info(f"Analyzing {len(image_paths)} Egyptian images...")
        results = self.egyptian_analyzer.analyze_images(image_paths)
        return results
    
    def synthesize_knowledge(self, data_files):
        """Synthesize knowledge from analysis data."""
        self.logger.info(f"Synthesizing knowledge from {len(data_files)} data files...")
        synthesis = self.aletheia_ai.synthesize(data_files)
        return synthesis
    
    def query(self, query_text):
        """Query the Aletheia AI with geometric intent."""
        self.logger.info(f"Processing query: {query_text}")
        response = self.aletheia_ai.process_query(query_text)
        return response
    
    def interactive_mode(self):
        """Enter interactive mode."""
        self.logger.info("Entering interactive mode...")
        print("\n" + "="*80)
        print("ALETHEIA CQE OPERATING SYSTEM - Interactive Mode")
        print("="*80)
        print("\nCommands:")
        print("  analyze <path>  - Analyze Egyptian images")
        print("  query <text>    - Query the AI")
        print("  synthesize      - Synthesize all available data")
        print("  status          - Show system status")
        print("  help            - Show this help")
        print("  exit            - Exit interactive mode")
        print()
        
        while True:
            try:
                cmd = input("aletheia> ").strip()
                
                if not cmd:
                    continue
                elif cmd == "exit":
                    print("Exiting Aletheia...")
                    break
                elif cmd == "help":
                    print("Commands: analyze, query, synthesize, status, help, exit")
                elif cmd == "status":
                    self._show_status()
                elif cmd.startswith("query "):
                    query_text = cmd[6:]
                    response = self.query(query_text)
                    print(f"\n{response}\n")
                elif cmd.startswith("analyze "):
                    path = cmd[8:]
                    results = self.analyze_egyptian([path])
                    print(f"\nAnalysis complete: {results}\n")
                elif cmd == "synthesize":
                    synthesis = self.synthesize_knowledge([])
                    print(f"\nSynthesis complete: {synthesis}\n")
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_status(self):
        """Show system status."""
        print("\n" + "="*80)
        print("ALETHEIA CQE SYSTEM STATUS")
        print("="*80)
        print(f"Version: {__version__}")
        print(f"CQE Engine: {self.cqe_engine.status()}")
        print(f"Aletheia AI: {self.aletheia_ai.status()}")
        print(f"Egyptian Analyzer: {self.egyptian_analyzer.status()}")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aletheia CQE Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aletheia.py --mode interactive
  python aletheia.py --mode analyze --input images/
  python aletheia.py --mode query --text "Explain E8 projection"
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["analyze", "synthesize", "query", "interactive"],
        default="interactive",
        help="Operating mode"
    )
    
    parser.add_argument(
        "--input",
        help="Input file or directory"
    )
    
    parser.add_argument(
        "--text",
        help="Query text (for query mode)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file or directory"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Aletheia CQE System v{__version__}"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = AletheiaSystem(verbose=args.verbose)
    
    # Execute based on mode
    if args.mode == "interactive":
        system.interactive_mode()
    elif args.mode == "analyze":
        if not args.input:
            print("Error: --input required for analyze mode")
            sys.exit(1)
        results = system.analyze_egyptian([args.input])
        print(f"Analysis complete: {results}")
    elif args.mode == "query":
        if not args.text:
            print("Error: --text required for query mode")
            sys.exit(1)
        response = system.query(args.text)
        print(response)
    elif args.mode == "synthesize":
        synthesis = system.synthesize_knowledge([])
        print(f"Synthesis complete: {synthesis}")


if __name__ == "__main__":
    main()

