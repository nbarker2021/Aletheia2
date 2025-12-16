"""
Interface Manager - User-Facing Interfaces

Provides CLI, REST API, and Natural Language interfaces.
"""

import json
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from spine.kernel import CQEKernel, CQEAtom
from spine.io_manager import IOManager
from spine.governance import GovernanceEngine
from spine.reasoning import ReasoningEngine
from spine.storage import StorageManager
from spine.speedlight import SpeedLight, get_speedlight


@dataclass
class InterfaceResponse:
    """Standard response from any interface."""
    success: bool
    data: Any
    message: str
    receipts: int


class InterfaceManager:
    """
    Interface Manager - Unified interface for all user interactions.
    
    Supports:
    - CLI commands
    - REST-like request/response
    - Natural language queries
    """
    
    def __init__(
        self,
        kernel: Optional[CQEKernel] = None,
        io_manager: Optional[IOManager] = None,
        governance: Optional[GovernanceEngine] = None,
        reasoning: Optional[ReasoningEngine] = None,
        storage: Optional[StorageManager] = None,
        speedlight: Optional[SpeedLight] = None
    ):
        self.speedlight = speedlight or get_speedlight()
        self.kernel = kernel or CQEKernel(self.speedlight)
        self.io_manager = io_manager or IOManager()
        self.governance = governance or GovernanceEngine()
        self.reasoning = reasoning or ReasoningEngine()
        self.storage = storage or StorageManager()
    
    def cli(self, args: List[str]) -> InterfaceResponse:
        """
        Process CLI-style commands.
        
        Commands:
        - status: Get system status
        - ingest <data>: Ingest data as atom
        - process <atom_id> <slice>: Process atom through slice
        - query <vector>: Find similar atoms
        - help: Show help
        """
        if not args:
            return self._help()
        
        cmd = args[0].lower()
        cmd_args = args[1:]
        
        if cmd == "status":
            return self._status()
        elif cmd == "ingest":
            return self._ingest(" ".join(cmd_args))
        elif cmd == "process":
            if len(cmd_args) < 2:
                return InterfaceResponse(False, None, "Usage: process <atom_id> <slice>", 0)
            return self._process(cmd_args[0], cmd_args[1])
        elif cmd == "query":
            return self._query(" ".join(cmd_args))
        elif cmd == "help":
            return self._help()
        else:
            return InterfaceResponse(False, None, f"Unknown command: {cmd}", 0)
    
    def rest(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process REST-like requests.
        
        Request format:
        {
            "action": "status|ingest|process|query",
            "data": <action-specific data>,
            "options": <optional parameters>
        }
        """
        action = request.get("action", "status")
        data = request.get("data")
        options = request.get("options", {})
        
        if action == "status":
            response = self._status()
        elif action == "ingest":
            response = self._ingest(data, **options)
        elif action == "process":
            atom_id = data.get("atom_id") if isinstance(data, dict) else data
            slice_name = options.get("slice", "morsr")
            response = self._process(atom_id, slice_name)
        elif action == "query":
            response = self._query(data)
        else:
            response = InterfaceResponse(False, None, f"Unknown action: {action}", 0)
        
        return {
            "success": response.success,
            "data": response.data,
            "message": response.message,
            "receipts": response.receipts
        }
    
    def natural(self, text: str) -> InterfaceResponse:
        """
        Process natural language queries.
        
        Simple keyword-based routing for demonstration.
        """
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["status", "state", "how are you"]):
            return self._status()
        elif any(w in text_lower for w in ["ingest", "add", "store", "save"]):
            # Extract data after the keyword
            for keyword in ["ingest", "add", "store", "save"]:
                if keyword in text_lower:
                    idx = text_lower.index(keyword) + len(keyword)
                    data = text[idx:].strip()
                    if data:
                        return self._ingest(data)
            return InterfaceResponse(False, None, "What would you like to ingest?", 0)
        elif any(w in text_lower for w in ["find", "search", "query", "similar"]):
            # Extract query
            for keyword in ["find", "search", "query", "similar"]:
                if keyword in text_lower:
                    idx = text_lower.index(keyword) + len(keyword)
                    query = text[idx:].strip()
                    if query:
                        return self._query(query)
            return InterfaceResponse(False, None, "What would you like to find?", 0)
        elif any(w in text_lower for w in ["help", "what can you do"]):
            return self._help()
        else:
            # Default: try to ingest as data
            return self._ingest(text)
    
    def _status(self) -> InterfaceResponse:
        """Get system status."""
        status = {
            "kernel": self.kernel.get_status(),
            "governance": self.governance.get_status(),
            "reasoning": self.reasoning.get_status(),
            "storage": self.storage.get_status(),
            "speedlight": self.speedlight.get_summary()
        }
        return InterfaceResponse(
            success=True,
            data=status,
            message="System operational",
            receipts=len(self.speedlight.ledger)
        )
    
    def _ingest(self, data: Any, format: str = "auto") -> InterfaceResponse:
        """Ingest data as an atom."""
        try:
            atom = self.io_manager.ingest(data, format)
            
            # Check governance
            allowed, reason = self.governance.check(atom)
            if not allowed:
                return InterfaceResponse(
                    success=False,
                    data=None,
                    message=f"Governance rejected: {reason}",
                    receipts=len(self.speedlight.ledger)
                )
            
            # Store atom
            atom_id = self.storage.store(atom)
            
            return InterfaceResponse(
                success=True,
                data={"atom_id": atom_id, "atom": atom.to_dict()},
                message=f"Ingested and stored as {atom_id}",
                receipts=len(self.speedlight.ledger)
            )
        except Exception as e:
            return InterfaceResponse(
                success=False,
                data=None,
                message=f"Ingest failed: {str(e)}",
                receipts=len(self.speedlight.ledger)
            )
    
    def _process(self, atom_id: str, slice_name: str) -> InterfaceResponse:
        """Process an atom through a slice."""
        atom = self.storage.retrieve(atom_id)
        if atom is None:
            return InterfaceResponse(
                success=False,
                data=None,
                message=f"Atom not found: {atom_id}",
                receipts=len(self.speedlight.ledger)
            )
        
        result = self.reasoning.route(atom, slice_name)
        
        if result.success:
            # Store the result atom
            new_id = self.storage.store(result.atom)
            return InterfaceResponse(
                success=True,
                data={"atom_id": new_id, "result": result.metadata},
                message=result.message,
                receipts=len(self.speedlight.ledger)
            )
        else:
            return InterfaceResponse(
                success=False,
                data=None,
                message=result.message,
                receipts=len(self.speedlight.ledger)
            )
    
    def _query(self, query_text: str) -> InterfaceResponse:
        """Query for similar atoms."""
        # Convert query to atom
        query_atom = self.io_manager.ingest(query_text)
        
        # Find similar
        results = self.storage.query_by_atom(query_atom, k=5)
        
        return InterfaceResponse(
            success=True,
            data={"results": [{"id": r[0], "distance": r[1]} for r in results]},
            message=f"Found {len(results)} similar atoms",
            receipts=len(self.speedlight.ledger)
        )
    
    def _help(self) -> InterfaceResponse:
        """Show help."""
        help_text = """
Morphonic Operation Platform - Interface

CLI Commands:
  status              - Get system status
  ingest <data>       - Ingest data as atom
  process <id> <slice> - Process atom through slice
  query <text>        - Find similar atoms
  help                - Show this help

Available Slices:
  morsr    - Multi-Objective Randomized Search and Repair
  sacnum   - Sacred Numerology analysis
  spectral - Spectral/eigenvalue analysis
"""
        return InterfaceResponse(
            success=True,
            data={"help": help_text},
            message="Help displayed",
            receipts=len(self.speedlight.ledger)
        )
