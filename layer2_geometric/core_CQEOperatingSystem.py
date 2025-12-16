class CQEOperatingSystem:
    """Universal Operating System using CQE principles"""
    
    def __init__(self, config: CQEOSConfig = None):
        self.config = config or CQEOSConfig()
        self.state = CQEOSState.INITIALIZING
        self.start_time = time.time()
        
        # Core components
        self.kernel: Optional[CQEKernel] = None
        self.io_manager: Optional[CQEIOManager] = None
        self.governance_engine: Optional[CQEGovernanceEngine] = None
        self.language_engine: Optional[CQELanguageEngine] = None
        self.reasoning_engine: Optional[CQEReasoningEngine] = None
        self.storage_manager: Optional[CQEStorageManager] = None
        self.interface_manager: Optional[CQEInterfaceManager] = None
        
        # System state
        self.system_atoms: Dict[str, CQEAtom] = {}
        self.running_processes: Dict[str, threading.Thread] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue: List[Dict[str, Any]] = []
        
        # Logging
        self.logger = self._setup_logging()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("CQE Operating System initialized")
    
    def boot(self) -> bool:
        """Boot the CQE Operating System"""
        try:
            self.logger.info("Booting CQE Operating System...")
            self.state = CQEOSState.INITIALIZING
            
            # Create base directory
            os.makedirs(self.config.base_path, exist_ok=True)
            
            # Initialize core kernel
            self.logger.info("Initializing CQE Kernel...")
            self.kernel = CQEKernel()
            
            # Initialize storage manager
            self.logger.info("Initializing Storage Manager...")
            storage_config = StorageConfig(
                storage_type=self.config.storage_type,
                base_path=self.config.base_path,
                max_memory_size=self.config.max_memory_atoms,
                backup_enabled=self.config.enable_backup,
                backup_interval=self.config.backup_interval
            )
            self.storage_manager = CQEStorageManager(self.kernel, storage_config)
            
            # Connect storage to kernel
            self.kernel.memory_manager = self.storage_manager
            
            # Initialize governance engine
            self.logger.info("Initializing Governance Engine...")
            self.governance_engine = CQEGovernanceEngine(self.kernel)
            self.governance_engine.set_active_policy(self.config.governance_level.value)
            
            # Initialize language engine
            self.logger.info("Initializing Language Engine...")
            self.language_engine = CQELanguageEngine(self.kernel)
            
            # Initialize reasoning engine
            self.logger.info("Initializing Reasoning Engine...")
            self.reasoning_engine = CQEReasoningEngine(self.kernel)
            
            # Initialize I/O manager
            self.logger.info("Initializing I/O Manager...")
            self.io_manager = CQEIOManager(self.kernel)
            
            # Initialize interface manager
            self.logger.info("Initializing Interface Manager...")
            self.interface_manager = CQEInterfaceManager(self.kernel)
            
            # Connect components to kernel
            self.kernel.governance_engine = self.governance_engine
            self.kernel.language_engine = self.language_engine
            self.kernel.reasoning_engine = self.reasoning_engine
            self.kernel.io_manager = self.io_manager
            self.kernel.interface_manager = self.interface_manager
            
            # Register enabled interfaces
            for interface_type in self.config.enabled_interfaces:
                self.interface_manager.register_interface(interface_type)
            
            # Create system atoms
            self._create_system_atoms()
            
            # Start system processes
            self._start_system_processes()
            
            # Set state to running
            self.state = CQEOSState.RUNNING
            
            self.logger.info("CQE Operating System boot completed successfully")
            self._emit_event("system_booted", {"boot_time": time.time() - self.start_time})
            
            return True
        
        except Exception as e:
            self.logger.error(f"Boot failed: {e}")
            self.state = CQEOSState.ERROR
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the CQE Operating System"""
        try:
            self.logger.info("Shutting down CQE Operating System...")
            self.state = CQEOSState.SHUTTING_DOWN
            
            # Emit shutdown event
            self._emit_event("system_shutting_down", {})
            
            # Stop system processes
            self._stop_system_processes()
            
            # Backup data if enabled
            if self.config.enable_backup and self.storage_manager:
                self.logger.info("Creating final backup...")
                self.storage_manager.backup_storage()
            
            # Shutdown components in reverse order
            if self.interface_manager:
                self.logger.info("Shutting down Interface Manager...")
                # Interface manager shutdown logic
            
            if self.io_manager:
                self.logger.info("Shutting down I/O Manager...")
                # I/O manager shutdown logic
            
            if self.reasoning_engine:
                self.logger.info("Shutting down Reasoning Engine...")
                # Reasoning engine shutdown logic
            
            if self.language_engine:
                self.logger.info("Shutting down Language Engine...")
                # Language engine shutdown logic
            
            if self.governance_engine:
                self.logger.info("Shutting down Governance Engine...")
                # Governance engine shutdown logic
            
            if self.storage_manager:
                self.logger.info("Shutting down Storage Manager...")
                # Storage manager shutdown logic
            
            if self.kernel:
                self.logger.info("Shutting down Kernel...")
                # Kernel shutdown logic
            
            self.state = CQEOSState.STOPPED
            self.logger.info("CQE Operating System shutdown completed")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            self.state = CQEOSState.ERROR
            return False
    
    def execute_operation(self, operation: CQEOperationType, data: Any, 
                         parameters: Dict[str, Any] = None) -> str:
        """Execute a CQE operation"""
        if self.state != CQEOSState.RUNNING:
            raise RuntimeError(f"Cannot execute operation in state: {self.state}")
        
        if not self.kernel:
            raise RuntimeError("Kernel not initialized")
        
        # Create operation atom
        operation_atom = CQEAtom(
            data={
                'operation': operation.value,
                'data': data,
                'parameters': parameters or {},
                'timestamp': time.time()
            },
            metadata={'system_operation': True, 'operation_type': operation.value}
        )
        
        # Execute through kernel
        result_atom_id = self.kernel.execute_operation(operation, operation_atom)
        
        # Log operation
        self.logger.debug(f"Executed operation {operation.value}: {result_atom_id}")
        
        return result_atom_id
    
    def process_input(self, input_data: Any, interface_type: InterfaceType = InterfaceType.CQE_NATIVE,
                     user_id: str = None, session_id: str = None) -> str:
        """Process input through the appropriate interface"""
        if not self.interface_manager:
            raise RuntimeError("Interface manager not initialized")
        
        # Create interface request
        from .interface.cqe_interface_manager import InterfaceRequest, InteractionMode
        
        request = InterfaceRequest(
            request_id=f"req_{int(time.time() * 1000000)}",
            interface_type=interface_type,
            interaction_mode=InteractionMode.SYNCHRONOUS,
            content=input_data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Process request
        response_id = self.interface_manager.process_request(request)
        
        return response_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'state': self.state.value,
            'uptime': time.time() - self.start_time,
            'components': {},
            'metrics': self.system_metrics,
            'config': {
                'base_path': self.config.base_path,
                'storage_type': self.config.storage_type.value,
                'governance_level': self.config.governance_level.value,
                'enabled_interfaces': [iface.value for iface in self.config.enabled_interfaces]
            }
        }
        
        # Component status
        if self.kernel:
            status['components']['kernel'] = {'status': 'running'}
        
        if self.storage_manager:
            status['components']['storage'] = self.storage_manager.get_storage_statistics().__dict__
        
        if self.governance_engine:
            status['components']['governance'] = self.governance_engine.get_governance_status()
        
        if self.interface_manager:
            status['components']['interface'] = self.interface_manager.get_interface_status()
        
        return status
    
    def create_session(self, user_id: str, interface_type: InterfaceType = InterfaceType.CQE_NATIVE,
                      preferences: Dict[str, Any] = None) -> str:
        """Create a new user session"""
        if not self.interface_manager:
            raise RuntimeError("Interface manager not initialized")
        
        session_id = self.interface_manager.create_session(user_id, interface_type, preferences)
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        self._emit_event("session_created", {
            'session_id': session_id,
            'user_id': user_id,
            'interface_type': interface_type.value
        })
        
        return session_id
    
    def query_data(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query data from the system"""
        if not self.storage_manager:
            raise RuntimeError("Storage manager not initialized")
        
        atoms = self.storage_manager.query_atoms(query, limit)
        return [atom.to_dict() for atom in atoms]
    
    def reason_about(self, goal: str, reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE) -> Dict[str, Any]:
        """Perform reasoning about a goal"""
        if not self.reasoning_engine:
            raise RuntimeError("Reasoning engine not initialized")
        
        chain_id = self.reasoning_engine.reason(goal, reasoning_type)
        explanation = self.reasoning_engine.generate_explanation(goal, chain_id)
        
        return {
            'goal': goal,
            'reasoning_type': reasoning_type.value,
            'chain_id': chain_id,
            'explanation': explanation
        }
    
    def process_language(self, text: str, language_hint: str = None) -> List[str]:
        """Process natural language text"""
        if not self.language_engine:
            raise RuntimeError("Language engine not initialized")
        
        atom_ids = self.language_engine.process_text(text, language_hint)
        return atom_ids
    
    def ingest_data(self, source_type: str, location: str, format: str = None) -> List[str]:
        """Ingest data from external source"""
        if not self.io_manager:
            raise RuntimeError("I/O manager not initialized")
        
        source_id = self.io_manager.register_data_source(source_type, location, format)
        atom_ids = self.io_manager.ingest_data(source_id)
        
        return atom_ids
    
    def export_data(self, atom_ids: List[str], output_format: str, output_location: str) -> bool:
        """Export data to external format"""
        if not self.io_manager:
            raise RuntimeError("I/O manager not initialized")
        
        return self.io_manager.export_data(atom_ids, output_format, output_location)
    
    def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance"""
        optimization_results = {
            'storage_optimization': {},
            'governance_optimization': {},
            'performance_improvement': {}
        }
        
        # Optimize storage
        if self.storage_manager:
            storage_results = self.storage_manager.optimize_storage()
            optimization_results['storage_optimization'] = storage_results
        
        # Optimize governance
        if self.governance_engine:
            # Governance optimization logic
            pass
        
        # Update metrics
        self._update_system_metrics()
        
        self.logger.info("System optimization completed")
        self._emit_event("system_optimized", optimization_results)
        
        return optimization_results
    
    def backup_system(self, backup_path: str = None) -> bool:
        """Create system backup"""
        if not self.storage_manager:
            return False
        
        success = self.storage_manager.backup_storage(backup_path)
        
        if success:
            self.logger.info(f"System backup created: {backup_path}")
            self._emit_event("system_backed_up", {'backup_path': backup_path})
        else:
            self.logger.error("System backup failed")
        
        return success
    
    def restore_system(self, backup_path: str) -> bool:
        """Restore system from backup"""
        if not self.storage_manager:
            return False
        
        success = self.storage_manager.restore_from_backup(backup_path)
        
        if success:
            self.logger.info(f"System restored from: {backup_path}")
            self._emit_event("system_restored", {'backup_path': backup_path})
        else:
            self.logger.error("System restore failed")
        
        return success
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    def run_interactive_shell(self):
        """Run interactive command shell"""
        print("CQE Operating System Interactive Shell")
        print("Type 'help' for available commands, 'exit' to quit")
        
        while self.state == CQEOSState.RUNNING:
            try:
                command = input("cqe> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['exit', 'quit']:
                    break
                
                # Process command through interface manager
                response_id = self.process_input(command, InterfaceType.COMMAND_LINE)
                
                # Get and display response
                if self.interface_manager:
                    response = self.interface_manager.get_response(response_id)
                    if response:
                        if isinstance(response.content, str):
                            print(response.content)
                        else:
                            print(json.dumps(response.content, indent=2))
                    else:
                        print("No response received")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Exiting CQE Operating System")
    
    def run_daemon(self):
        """Run as daemon process"""
        self.logger.info("Running CQE OS as daemon")
        
        try:
            while self.state == CQEOSState.RUNNING:
                # Perform periodic maintenance
                self._perform_maintenance()
                
                # Process events
                self._process_events()
                
                # Sleep briefly
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            self.logger.info("Daemon interrupted")
        except Exception as e:
            self.logger.error(f"Daemon error: {e}")
            self.state = CQEOSState.ERROR
    
    # Private Methods
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('cqe_os')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        
        if signum in [signal.SIGINT, signal.SIGTERM]:
            self.shutdown()
            sys.exit(0)
    
    def _create_system_atoms(self):
        """Create fundamental system atoms"""
        if not self.kernel:
            return
        
        # System configuration atom
        config_atom = CQEAtom(
            data={
                'type': 'system_config',
                'config': self.config.__dict__,
                'boot_time': self.start_time
            },
            metadata={'system_atom': True, 'atom_type': 'config'}
        )
        
        config_atom_id = self.kernel.memory_manager.store_atom(config_atom)
        self.system_atoms['config'] = config_atom
        
        # System status atom
        status_atom = CQEAtom(
            data={
                'type': 'system_status',
                'state': self.state.value,
                'uptime': 0
            },
            metadata={'system_atom': True, 'atom_type': 'status'}
        )
        
        status_atom_id = self.kernel.memory_manager.store_atom(status_atom)
        self.system_atoms['status'] = status_atom
        
        self.logger.debug("System atoms created")
    
    def _start_system_processes(self):
        """Start background system processes"""
        # Metrics collection process
        if self.config.enable_monitoring:
            metrics_thread = threading.Thread(target=self._metrics_collector, daemon=True)
            metrics_thread.start()
            self.running_processes['metrics'] = metrics_thread
        
        # Backup process
        if self.config.enable_backup:
            backup_thread = threading.Thread(target=self._backup_scheduler, daemon=True)
            backup_thread.start()
            self.running_processes['backup'] = backup_thread
        
        # Governance enforcement process
        governance_thread = threading.Thread(target=self._governance_enforcer, daemon=True)
        governance_thread.start()
        self.running_processes['governance'] = governance_thread
        
        self.logger.debug("System processes started")
    
    def _stop_system_processes(self):
        """Stop background system processes"""
        for process_name, thread in self.running_processes.items():
            self.logger.debug(f"Stopping process: {process_name}")
            # Threads are daemon threads, they will stop when main thread exits
        
        self.running_processes.clear()
    
    def _metrics_collector(self):
        """Background process to collect system metrics"""
        while self.state == CQEOSState.RUNNING:
            try:
                self._update_system_metrics()
                time.sleep(60)  # Collect metrics every minute
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(60)
    
    def _backup_scheduler(self):
        """Background process to schedule backups"""
        last_backup = time.time()
        
        while self.state == CQEOSState.RUNNING:
            try:
                current_time = time.time()
                
                if current_time - last_backup >= self.config.backup_interval:
                    self.backup_system()
                    last_backup = current_time
                
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Backup scheduler error: {e}")
                time.sleep(300)
    
    def _governance_enforcer(self):
        """Background process to enforce governance"""
        while self.state == CQEOSState.RUNNING:
            try:
                if self.governance_engine and self.storage_manager:
                    # Get all atom IDs
                    atom_ids = self.storage_manager._get_all_atom_ids()
                    
                    # Enforce governance on a subset
                    batch_size = 100
                    for i in range(0, len(atom_ids), batch_size):
                        batch = atom_ids[i:i+batch_size]
                        self.governance_engine.enforce_governance(batch)
                
                time.sleep(300)  # Enforce every 5 minutes
            except Exception as e:
                self.logger.error(f"Governance enforcement error: {e}")
                time.sleep(300)
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        current_time = time.time()
        
        self.system_metrics.update({
            'timestamp': current_time,
            'uptime': current_time - self.start_time,
            'state': self.state.value,
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage(),
            'disk_usage': self._get_disk_usage(),
            'active_sessions': len(self.interface_manager.sessions) if self.interface_manager else 0,
            'total_atoms': len(self.storage_manager._get_all_atom_ids()) if self.storage_manager else 0
        })
        
        # Update system status atom
        if 'status' in self.system_atoms:
            status_atom = self.system_atoms['status']
            status_atom.data.update({
                'state': self.state.value,
                'uptime': current_time - self.start_time,
                'last_update': current_time
            })
            
            if self.kernel:
                self.kernel.memory_manager.store_atom(status_atom)
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _get_cpu_usage(self) -> Dict[str, Any]:
        """Get CPU usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.config.base_path)
            
            return {
                'total': total,
                'used': used,
                'free': free,
                'percent': (used / total) * 100
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a system event"""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        
        self.event_queue.append(event)
        
        # Call registered handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    def _process_events(self):
        """Process queued events"""
        while self.event_queue:
            event = self.event_queue.pop(0)
            self.logger.debug(f"Processing event: {event['type']}")
            
            # Create event atom
            if self.kernel:
                event_atom = CQEAtom(
                    data=event,
                    metadata={'system_event': True, 'event_type': event['type']}
                )
                self.kernel.memory_manager.store_atom(event_atom)
    
    def _perform_maintenance(self):
        """Perform periodic system maintenance"""
        # Optimize storage periodically
        if hasattr(self, '_last_optimization'):
            if time.time() - self._last_optimization > 3600:  # Every hour
                self.optimize_system()
                self._last_optimization = time.time()
        else:
            self._last_optimization = time.time()
        
        # Clean up old events
        if len(self.event_queue) > 1000:
            self.event_queue = self.event_queue[-500:]  # Keep last 500 events

# Convenience functions for easy usage