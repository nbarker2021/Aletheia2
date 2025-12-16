#!/usr/bin/env python3
"""
CQE Interface Manager
Universal user interface using CQE principles
"""

import json
import time
import asyncio
from typing import Any, Dict, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import queue
import hashlib
import re

from ..core.cqe_os_kernel import CQEAtom, CQEKernel, CQEOperationType

class InterfaceType(Enum):
    """Types of interfaces supported"""
    COMMAND_LINE = "command_line"
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    NATURAL_LANGUAGE = "natural_language"
    VISUAL = "visual"
    VOICE = "voice"
    GESTURE = "gesture"
    BRAIN_COMPUTER = "brain_computer"
    CQE_NATIVE = "cqe_native"

class InteractionMode(Enum):
    """Modes of interaction"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"
    REAL_TIME = "real_time"
    CONVERSATIONAL = "conversational"

class ResponseFormat(Enum):
    """Response format types"""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    BINARY = "binary"
    CQE_NATIVE = "cqe_native"

@dataclass
class InterfaceRequest:
    """Represents a request to the CQE system"""
    request_id: str
    interface_type: InterfaceType
    interaction_mode: InteractionMode
    content: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterfaceResponse:
    """Represents a response from the CQE system"""
    response_id: str
    request_id: str
    status: str  # success, error, partial, pending
    content: Any
    format: ResponseFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    processing_time: float = 0.0
    confidence: float = 1.0

@dataclass
class UserSession:
    """Represents a user session"""
    session_id: str
    user_id: str
    interface_type: InterfaceType
    start_time: float
    last_activity: float
    context: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # Request IDs
    active: bool = True

class CQEInterfaceManager:
    """Universal interface manager using CQE principles"""
    
    def __init__(self, kernel: CQEKernel):
        self.kernel = kernel
        self.interface_handlers: Dict[InterfaceType, Callable] = {}
        self.response_formatters: Dict[ResponseFormat, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Session management
        self.sessions: Dict[str, UserSession] = {}
        self.requests: Dict[str, InterfaceRequest] = {}
        self.responses: Dict[str, InterfaceResponse] = {}
        
        # Request processing
        self.request_queue = queue.Queue()
        self.response_cache: Dict[str, InterfaceResponse] = {}
        self.processing_threads: List[threading.Thread] = []
        
        # Interface state
        self.active_interfaces: Set[InterfaceType] = set()
        self.interface_configs: Dict[InterfaceType, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Initialize interface components
        self._initialize_interface_handlers()
        self._initialize_response_formatters()
        self._initialize_middleware()
        self._start_processing_threads()
    
    def _initialize_interface_handlers(self):
        """Initialize handlers for different interface types"""
        self.interface_handlers = {
            InterfaceType.COMMAND_LINE: self._handle_command_line,
            InterfaceType.REST_API: self._handle_rest_api,
            InterfaceType.GRAPHQL: self._handle_graphql,
            InterfaceType.WEBSOCKET: self._handle_websocket,
            InterfaceType.NATURAL_LANGUAGE: self._handle_natural_language,
            InterfaceType.VISUAL: self._handle_visual,
            InterfaceType.VOICE: self._handle_voice,
            InterfaceType.GESTURE: self._handle_gesture,
            InterfaceType.BRAIN_COMPUTER: self._handle_brain_computer,
            InterfaceType.CQE_NATIVE: self._handle_cqe_native
        }
    
    def _initialize_response_formatters(self):
        """Initialize response formatters"""
        self.response_formatters = {
            ResponseFormat.JSON: self._format_as_json,
            ResponseFormat.XML: self._format_as_xml,
            ResponseFormat.YAML: self._format_as_yaml,
            ResponseFormat.TEXT: self._format_as_text,
            ResponseFormat.HTML: self._format_as_html,
            ResponseFormat.MARKDOWN: self._format_as_markdown,
            ResponseFormat.BINARY: self._format_as_binary,
            ResponseFormat.CQE_NATIVE: self._format_as_cqe_native
        }
    
    def _initialize_middleware(self):
        """Initialize middleware for request/response processing"""
        self.middleware = [
            self._authentication_middleware,
            self._authorization_middleware,
            self._rate_limiting_middleware,
            self._validation_middleware,
            self._logging_middleware,
            self._caching_middleware,
            self._compression_middleware
        ]
    
    def _start_processing_threads(self):
        """Start background threads for request processing"""
        for i in range(4):  # 4 processing threads
            thread = threading.Thread(target=self._process_requests, daemon=True)
            thread.start()
            self.processing_threads.append(thread)
    
    def create_session(self, user_id: str, interface_type: InterfaceType,
                      preferences: Dict[str, Any] = None) -> str:
        """Create a new user session"""
        session_id = hashlib.md5(f"{user_id}:{interface_type.value}:{time.time()}".encode()).hexdigest()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            interface_type=interface_type,
            start_time=time.time(),
            last_activity=time.time(),
            preferences=preferences or {}
        )
        
        self.sessions[session_id] = session
        
        # Create session atom
        session_atom = CQEAtom(
            data={
                'session_id': session_id,
                'user_id': user_id,
                'interface_type': interface_type.value,
                'start_time': session.start_time
            },
            metadata={'interface_manager': True, 'user_session': True}
        )
        
        self.kernel.memory_manager.store_atom(session_atom)
        
        return session_id
    
    def process_request(self, request: InterfaceRequest) -> str:
        """Process an interface request"""
        # Apply middleware
        for middleware in self.middleware:
            request = middleware(request, 'request')
            if not request:  # Middleware rejected request
                return self._create_error_response("Request rejected by middleware")
        
        # Store request
        self.requests[request.request_id] = request
        
        # Update session activity
        if request.session_id and request.session_id in self.sessions:
            session = self.sessions[request.session_id]
            session.last_activity = time.time()
            session.history.append(request.request_id)
        
        # Queue for processing
        if request.interaction_mode == InteractionMode.SYNCHRONOUS:
            # Process immediately
            response = self._process_request_sync(request)
            return response.response_id
        else:
            # Queue for async processing
            self.request_queue.put(request)
            
            # Create pending response
            response = InterfaceResponse(
                response_id=hashlib.md5(f"response:{request.request_id}:{time.time()}".encode()).hexdigest(),
                request_id=request.request_id,
                status="pending",
                content={"message": "Request queued for processing"},
                format=ResponseFormat.JSON
            )
            
            self.responses[response.response_id] = response
            return response.response_id
    
    def get_response(self, response_id: str) -> Optional[InterfaceResponse]:
        """Get a response by ID"""
        return self.responses.get(response_id)
    
    def stream_response(self, response_id: str) -> Iterator[Dict[str, Any]]:
        """Stream response data for real-time interfaces"""
        response = self.responses.get(response_id)
        if not response:
            yield {"error": "Response not found"}
            return
        
        if response.status == "pending":
            yield {"status": "pending", "message": "Processing request..."}
            
            # Wait for completion (simplified)
            while response.status == "pending":
                time.sleep(0.1)
                response = self.responses.get(response_id)
                if not response:
                    break
        
        if response:
            yield {
                "status": response.status,
                "content": response.content,
                "metadata": response.metadata
            }
    
    def register_interface(self, interface_type: InterfaceType, 
                          config: Dict[str, Any] = None) -> bool:
        """Register and activate an interface type"""
        try:
            self.active_interfaces.add(interface_type)
            self.interface_configs[interface_type] = config or {}
            
            # Initialize interface-specific components
            if interface_type == InterfaceType.REST_API:
                self._initialize_rest_api(config)
            elif interface_type == InterfaceType.WEBSOCKET:
                self._initialize_websocket(config)
            elif interface_type == InterfaceType.NATURAL_LANGUAGE:
                self._initialize_natural_language(config)
            
            return True
        
        except Exception as e:
            print(f"Interface registration error: {e}")
            return False
    
    def unregister_interface(self, interface_type: InterfaceType) -> bool:
        """Unregister and deactivate an interface type"""
        try:
            self.active_interfaces.discard(interface_type)
            if interface_type in self.interface_configs:
                del self.interface_configs[interface_type]
            
            return True
        
        except Exception as e:
            print(f"Interface unregistration error: {e}")
            return False
    
    def get_interface_status(self) -> Dict[str, Any]:
        """Get status of all interfaces"""
        return {
            'active_interfaces': [iface.value for iface in self.active_interfaces],
            'total_sessions': len(self.sessions),
            'active_sessions': len([s for s in self.sessions.values() if s.active]),
            'pending_requests': self.request_queue.qsize(),
            'total_requests': len(self.requests),
            'total_responses': len(self.responses),
            'performance_metrics': dict(self.performance_metrics),
            'error_counts': dict(self.error_counts)
        }
    
    # Request Processing
    def _process_requests(self):
        """Background thread for processing requests"""
        while True:
            try:
                request = self.request_queue.get(timeout=1.0)
                response = self._process_request_sync(request)
                
                # Update response in storage
                self.responses[response.response_id] = response
                
                self.request_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Request processing error: {e}")
    
    def _process_request_sync(self, request: InterfaceRequest) -> InterfaceResponse:
        """Process a request synchronously"""
        start_time = time.time()
        
        try:
            # Get appropriate handler
            handler = self.interface_handlers.get(request.interface_type, self._handle_generic)
            
            # Process request
            result = handler(request)
            
            # Determine response format
            response_format = self._determine_response_format(request)
            
            # Format response
            formatter = self.response_formatters.get(response_format, self._format_as_json)
            formatted_content = formatter(result)
            
            # Create response
            response = InterfaceResponse(
                response_id=hashlib.md5(f"response:{request.request_id}:{time.time()}".encode()).hexdigest(),
                request_id=request.request_id,
                status="success",
                content=formatted_content,
                format=response_format,
                processing_time=time.time() - start_time,
                confidence=result.get('confidence', 1.0) if isinstance(result, dict) else 1.0
            )
            
            # Apply response middleware
            for middleware in reversed(self.middleware):
                response = middleware(response, 'response')
                if not response:
                    break
            
            # Update performance metrics
            self.performance_metrics[request.interface_type.value].append(response.processing_time)
            
            return response
        
        except Exception as e:
            # Create error response
            error_response = InterfaceResponse(
                response_id=hashlib.md5(f"error:{request.request_id}:{time.time()}".encode()).hexdigest(),
                request_id=request.request_id,
                status="error",
                content={"error": str(e), "type": type(e).__name__},
                format=ResponseFormat.JSON,
                processing_time=time.time() - start_time
            )
            
            # Update error counts
            self.error_counts[request.interface_type.value] += 1
            
            return error_response
    
    # Interface Handlers
    def _handle_command_line(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle command line interface requests"""
        command = request.content
        
        if isinstance(command, str):
            # Parse command
            parts = command.strip().split()
            if not parts:
                return {"error": "Empty command"}
            
            cmd = parts[0].lower()
            args = parts[1:]
            
            # Execute command
            if cmd == "help":
                return self._get_help_content()
            elif cmd == "status":
                return self.get_interface_status()
            elif cmd == "query":
                return self._execute_query(args)
            elif cmd == "create":
                return self._create_atom(args)
            elif cmd == "reason":
                return self._execute_reasoning(args)
            else:
                return {"error": f"Unknown command: {cmd}"}
        
        return {"error": "Invalid command format"}
    
    def _handle_rest_api(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle REST API requests"""
        method = request.parameters.get('method', 'GET')
        path = request.parameters.get('path', '/')
        
        if method == 'GET':
            if path.startswith('/atoms'):
                return self._api_get_atoms(request)
            elif path.startswith('/sessions'):
                return self._api_get_sessions(request)
            elif path.startswith('/status'):
                return self.get_interface_status()
        
        elif method == 'POST':
            if path.startswith('/atoms'):
                return self._api_create_atom(request)
            elif path.startswith('/query'):
                return self._api_query(request)
            elif path.startswith('/reason'):
                return self._api_reason(request)
        
        return {"error": "API endpoint not found"}
    
    def _handle_graphql(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle GraphQL requests"""
        query = request.content.get('query', '')
        variables = request.content.get('variables', {})
        
        # Simple GraphQL parsing (would use proper parser in production)
        if 'atoms' in query:
            return self._graphql_atoms(query, variables)
        elif 'sessions' in query:
            return self._graphql_sessions(query, variables)
        
        return {"error": "GraphQL query not supported"}
    
    def _handle_websocket(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle WebSocket requests"""
        message_type = request.parameters.get('type', 'message')
        
        if message_type == 'subscribe':
            return self._websocket_subscribe(request)
        elif message_type == 'unsubscribe':
            return self._websocket_unsubscribe(request)
        elif message_type == 'message':
            return self._websocket_message(request)
        
        return {"error": "Unknown WebSocket message type"}
    
    def _handle_natural_language(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle natural language requests"""
        text = request.content
        
        if not isinstance(text, str):
            return {"error": "Natural language input must be text"}
        
        # Process through language engine
        language_engine = self.kernel.language_engine
        atom_ids = language_engine.process_text(text)
        
        # Extract intent and entities
        intent = self._extract_intent(text)
        entities = self._extract_entities(text)
        
        # Execute based on intent
        if intent == 'query':
            return self._execute_natural_query(text, entities)
        elif intent == 'create':
            return self._create_from_natural_language(text, entities)
        elif intent == 'reason':
            return self._reason_from_natural_language(text, entities)
        else:
            return {
                "response": f"I understand you said: '{text}'",
                "intent": intent,
                "entities": entities,
                "processed_atoms": atom_ids
            }
    
    def _handle_visual(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle visual interface requests"""
        # Placeholder for visual interface handling
        return {"message": "Visual interface processing not implemented"}
    
    def _handle_voice(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle voice interface requests"""
        # Placeholder for voice interface handling
        return {"message": "Voice interface processing not implemented"}
    
    def _handle_gesture(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle gesture interface requests"""
        # Placeholder for gesture interface handling
        return {"message": "Gesture interface processing not implemented"}
    
    def _handle_brain_computer(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle brain-computer interface requests"""
        # Placeholder for BCI handling
        return {"message": "Brain-computer interface processing not implemented"}
    
    def _handle_cqe_native(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle CQE native interface requests"""
        if isinstance(request.content, dict) and 'operation' in request.content:
            operation = request.content['operation']
            
            if operation == 'create_atom':
                return self._cqe_create_atom(request.content)
            elif operation == 'query_atoms':
                return self._cqe_query_atoms(request.content)
            elif operation == 'reason':
                return self._cqe_reason(request.content)
            elif operation == 'transform':
                return self._cqe_transform(request.content)
        
        return {"error": "Invalid CQE native request"}
    
    def _handle_generic(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Generic handler for unknown interface types"""
        return {
            "message": f"Generic handling for {request.interface_type.value}",
            "content": request.content,
            "parameters": request.parameters
        }
    
    # Response Formatters
    def _format_as_json(self, content: Any) -> str:
        """Format response as JSON"""
        return json.dumps(content, default=str, indent=2)
    
    def _format_as_xml(self, content: Any) -> str:
        """Format response as XML"""
        # Simple XML formatting
        if isinstance(content, dict):
            xml_parts = ["<response>"]
            for key, value in content.items():
                xml_parts.append(f"<{key}>{value}</{key}>")
            xml_parts.append("</response>")
            return '\n'.join(xml_parts)
        else:
            return f"<response>{content}</response>"
    
    def _format_as_yaml(self, content: Any) -> str:
        """Format response as YAML"""
        import yaml
        return yaml.dump(content, default_flow_style=False)
    
    def _format_as_text(self, content: Any) -> str:
        """Format response as plain text"""
        if isinstance(content, dict):
            lines = []
            for key, value in content.items():
                lines.append(f"{key}: {value}")
            return '\n'.join(lines)
        else:
            return str(content)
    
    def _format_as_html(self, content: Any) -> str:
        """Format response as HTML"""
        html_parts = ["<html><body>"]
        
        if isinstance(content, dict):
            html_parts.append("<dl>")
            for key, value in content.items():
                html_parts.append(f"<dt>{key}</dt><dd>{value}</dd>")
            html_parts.append("</dl>")
        else:
            html_parts.append(f"<p>{content}</p>")
        
        html_parts.append("</body></html>")
        return '\n'.join(html_parts)
    
    def _format_as_markdown(self, content: Any) -> str:
        """Format response as Markdown"""
        if isinstance(content, dict):
            lines = ["# Response", ""]
            for key, value in content.items():
                lines.append(f"**{key}:** {value}")
                lines.append("")
            return '\n'.join(lines)
        else:
            return f"# Response\n\n{content}"
    
    def _format_as_binary(self, content: Any) -> bytes:
        """Format response as binary"""
        import pickle
        return pickle.dumps(content)
    
    def _format_as_cqe_native(self, content: Any) -> Dict[str, Any]:
        """Format response in CQE native format"""
        return {
            "cqe_response": True,
            "content": content,
            "timestamp": time.time(),
            "format": "cqe_native"
        }
    
    # Middleware Functions
    def _authentication_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                                 direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Authentication middleware"""
        if direction == 'request' and isinstance(item, InterfaceRequest):
            # Check authentication
            if item.user_id is None and item.interface_type != InterfaceType.COMMAND_LINE:
                return None  # Reject unauthenticated requests
        
        return item
    
    def _authorization_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                                direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Authorization middleware"""
        # Placeholder for authorization logic
        return item
    
    def _rate_limiting_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                                direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Rate limiting middleware"""
        # Placeholder for rate limiting logic
        return item
    
    def _validation_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                             direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Validation middleware"""
        if direction == 'request' and isinstance(item, InterfaceRequest):
            # Validate request structure
            if not item.content:
                return None
        
        return item
    
    def _logging_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                          direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Logging middleware"""
        # Log requests and responses
        if direction == 'request' and isinstance(item, InterfaceRequest):
            print(f"Request: {item.interface_type.value} - {item.request_id}")
        elif direction == 'response' and isinstance(item, InterfaceResponse):
            print(f"Response: {item.status} - {item.response_id}")
        
        return item
    
    def _caching_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                          direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Caching middleware"""
        # Placeholder for caching logic
        return item
    
    def _compression_middleware(self, item: Union[InterfaceRequest, InterfaceResponse], 
                              direction: str) -> Union[InterfaceRequest, InterfaceResponse, None]:
        """Compression middleware"""
        # Placeholder for compression logic
        return item
    
    # Utility Methods
    def _determine_response_format(self, request: InterfaceRequest) -> ResponseFormat:
        """Determine appropriate response format"""
        # Check request preferences
        if 'format' in request.parameters:
            format_str = request.parameters['format'].lower()
            for fmt in ResponseFormat:
                if fmt.value == format_str:
                    return fmt
        
        # Default based on interface type
        if request.interface_type == InterfaceType.REST_API:
            return ResponseFormat.JSON
        elif request.interface_type == InterfaceType.COMMAND_LINE:
            return ResponseFormat.TEXT
        elif request.interface_type == InterfaceType.NATURAL_LANGUAGE:
            return ResponseFormat.TEXT
        elif request.interface_type == InterfaceType.CQE_NATIVE:
            return ResponseFormat.CQE_NATIVE
        else:
            return ResponseFormat.JSON
    
    def _create_error_response(self, error_message: str) -> str:
        """Create an error response"""
        response = InterfaceResponse(
            response_id=hashlib.md5(f"error:{time.time()}".encode()).hexdigest(),
            request_id="unknown",
            status="error",
            content={"error": error_message},
            format=ResponseFormat.JSON
        )
        
        self.responses[response.response_id] = response
        return response.response_id
    
    # Command Implementations
    def _get_help_content(self) -> Dict[str, Any]:
        """Get help content"""
        return {
            "commands": {
                "help": "Show this help message",
                "status": "Show system status",
                "query <criteria>": "Query atoms",
                "create <data>": "Create new atom",
                "reason <goal>": "Perform reasoning"
            },
            "interfaces": [iface.value for iface in self.active_interfaces]
        }
    
    def _execute_query(self, args: List[str]) -> Dict[str, Any]:
        """Execute a query command"""
        # Simple query parsing
        query = {}
        if args:
            query_str = ' '.join(args)
            # Parse simple key:value queries
            for part in query_str.split(','):
                if ':' in part:
                    key, value = part.split(':', 1)
                    query[key.strip()] = value.strip()
        
        # Execute query through storage manager
        atoms = self.kernel.memory_manager.query_atoms(query)
        
        return {
            "query": query,
            "results": len(atoms),
            "atoms": [atom.to_dict() for atom in atoms[:10]]  # Limit results
        }
    
    def _create_atom(self, args: List[str]) -> Dict[str, Any]:
        """Create a new atom"""
        if not args:
            return {"error": "No data provided"}
        
        data_str = ' '.join(args)
        
        # Try to parse as JSON, fallback to string
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            data = data_str
        
        # Create atom
        atom = CQEAtom(data=data, metadata={'created_via': 'command_line'})
        atom_id = self.kernel.memory_manager.store_atom(atom)
        
        return {
            "atom_id": atom_id,
            "data": data,
            "quad_encoding": atom.quad_encoding
        }
    
    def _execute_reasoning(self, args: List[str]) -> Dict[str, Any]:
        """Execute reasoning"""
        if not args:
            return {"error": "No goal provided"}
        
        goal = ' '.join(args)
        
        # Execute reasoning through reasoning engine
        reasoning_engine = self.kernel.reasoning_engine
        chain_id = reasoning_engine.reason(goal)
        
        return {
            "goal": goal,
            "reasoning_chain_id": chain_id,
            "explanation": reasoning_engine.generate_explanation(goal, chain_id)
        }
    
    # API Implementations
    def _api_get_atoms(self, request: InterfaceRequest) -> Dict[str, Any]:
        """API endpoint to get atoms"""
        limit = request.parameters.get('limit', 10)
        atoms = self.kernel.memory_manager.query_atoms({}, limit=limit)
        
        return {
            "atoms": [atom.to_dict() for atom in atoms],
            "count": len(atoms)
        }
    
    def _api_get_sessions(self, request: InterfaceRequest) -> Dict[str, Any]:
        """API endpoint to get sessions"""
        return {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "interface_type": session.interface_type.value,
                    "active": session.active,
                    "start_time": session.start_time,
                    "last_activity": session.last_activity
                }
                for session in self.sessions.values()
            ]
        }
    
    def _api_create_atom(self, request: InterfaceRequest) -> Dict[str, Any]:
        """API endpoint to create atom"""
        data = request.content
        atom = CQEAtom(data=data, metadata={'created_via': 'api'})
        atom_id = self.kernel.memory_manager.store_atom(atom)
        
        return {
            "atom_id": atom_id,
            "atom": atom.to_dict()
        }
    
    def _api_query(self, request: InterfaceRequest) -> Dict[str, Any]:
        """API endpoint for querying"""
        query = request.content.get('query', {})
        limit = request.content.get('limit', 10)
        
        atoms = self.kernel.memory_manager.query_atoms(query, limit=limit)
        
        return {
            "query": query,
            "results": [atom.to_dict() for atom in atoms],
            "count": len(atoms)
        }
    
    def _api_reason(self, request: InterfaceRequest) -> Dict[str, Any]:
        """API endpoint for reasoning"""
        goal = request.content.get('goal', '')
        reasoning_type = request.content.get('reasoning_type', 'deductive')
        
        reasoning_engine = self.kernel.reasoning_engine
        chain_id = reasoning_engine.reason(goal)
        
        return {
            "goal": goal,
            "reasoning_chain_id": chain_id,
            "explanation": reasoning_engine.generate_explanation(goal, chain_id)
        }
    
    # Natural Language Processing
    def _extract_intent(self, text: str) -> str:
        """Extract intent from natural language text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['find', 'search', 'query', 'get', 'show']):
            return 'query'
        elif any(word in text_lower for word in ['create', 'make', 'add', 'new']):
            return 'create'
        elif any(word in text_lower for word in ['reason', 'think', 'analyze', 'solve']):
            return 'reason'
        elif any(word in text_lower for word in ['help', 'assist', 'guide']):
            return 'help'
        else:
            return 'unknown'
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from natural language text"""
        entities = []
        
        # Simple entity extraction (would use NER in production)
        words = text.split()
        for word in words:
            if word.isdigit():
                entities.append({"type": "number", "value": word})
            elif word.startswith('@'):
                entities.append({"type": "user", "value": word[1:]})
            elif word.startswith('#'):
                entities.append({"type": "tag", "value": word[1:]})
        
        return entities
    
    def _execute_natural_query(self, text: str, entities: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute query from natural language"""
        # Convert natural language to query
        query = {}
        
        # Extract query criteria from entities
        for entity in entities:
            if entity["type"] == "tag":
                query["metadata.tags"] = entity["value"]
        
        atoms = self.kernel.memory_manager.query_atoms(query, limit=5)
        
        return {
            "natural_query": text,
            "extracted_query": query,
            "results": [atom.to_dict() for atom in atoms]
        }
    
    def _create_from_natural_language(self, text: str, entities: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create atom from natural language"""
        # Extract data from text
        data = {
            "natural_language_input": text,
            "extracted_entities": entities,
            "created_via": "natural_language"
        }
        
        atom = CQEAtom(data=data, metadata={'natural_language': True})
        atom_id = self.kernel.memory_manager.store_atom(atom)
        
        return {
            "atom_id": atom_id,
            "created_from": text,
            "atom": atom.to_dict()
        }
    
    def _reason_from_natural_language(self, text: str, entities: List[Dict[str, str]]) -> Dict[str, Any]:
        """Perform reasoning from natural language"""
        # Extract goal from text
        goal = text
        
        reasoning_engine = self.kernel.reasoning_engine
        chain_id = reasoning_engine.reason(goal)
        
        return {
            "natural_language_goal": text,
            "reasoning_chain_id": chain_id,
            "explanation": reasoning_engine.generate_explanation(goal, chain_id)
        }
    
    # Interface-specific initializers
    def _initialize_rest_api(self, config: Dict[str, Any]):
        """Initialize REST API interface"""
        # Placeholder for REST API initialization
        pass
    
    def _initialize_websocket(self, config: Dict[str, Any]):
        """Initialize WebSocket interface"""
        # Placeholder for WebSocket initialization
        pass
    
    def _initialize_natural_language(self, config: Dict[str, Any]):
        """Initialize natural language interface"""
        # Placeholder for NL interface initialization
        pass
    
    # WebSocket handlers
    def _websocket_subscribe(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle WebSocket subscription"""
        return {"message": "WebSocket subscription not implemented"}
    
    def _websocket_unsubscribe(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle WebSocket unsubscription"""
        return {"message": "WebSocket unsubscription not implemented"}
    
    def _websocket_message(self, request: InterfaceRequest) -> Dict[str, Any]:
        """Handle WebSocket message"""
        return {"message": "WebSocket message handling not implemented"}
    
    # GraphQL handlers
    def _graphql_atoms(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GraphQL atoms query"""
        return {"message": "GraphQL atoms query not implemented"}
    
    def _graphql_sessions(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GraphQL sessions query"""
        return {"message": "GraphQL sessions query not implemented"}
    
    # CQE Native handlers
    def _cqe_create_atom(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CQE native atom creation"""
        data = content.get('data', {})
        quad_encoding = content.get('quad_encoding')
        
        atom = CQEAtom(data=data, metadata={'created_via': 'cqe_native'})
        
        if quad_encoding:
            atom.quad_encoding = tuple(quad_encoding)
        
        atom_id = self.kernel.memory_manager.store_atom(atom)
        
        return {
            "atom_id": atom_id,
            "atom": atom.to_dict()
        }
    
    def _cqe_query_atoms(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CQE native atom query"""
        query = content.get('query', {})
        limit = content.get('limit', 10)
        
        atoms = self.kernel.memory_manager.query_atoms(query, limit=limit)
        
        return {
            "query": query,
            "results": [atom.to_dict() for atom in atoms]
        }
    
    def _cqe_reason(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CQE native reasoning"""
        goal = content.get('goal', '')
        reasoning_type = content.get('reasoning_type', 'deductive')
        
        reasoning_engine = self.kernel.reasoning_engine
        chain_id = reasoning_engine.reason(goal)
        
        return {
            "goal": goal,
            "reasoning_chain_id": chain_id,
            "explanation": reasoning_engine.generate_explanation(goal, chain_id)
        }
    
    def _cqe_transform(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CQE native transformation"""
        return {"message": "CQE transformation not implemented"}

# Export main classes
__all__ = [
    'CQEInterfaceManager', 'InterfaceRequest', 'InterfaceResponse', 'UserSession',
    'InterfaceType', 'InteractionMode', 'ResponseFormat'
]
