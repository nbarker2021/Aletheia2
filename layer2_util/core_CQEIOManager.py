class CQEIOManager:
    """Universal I/O Manager using CQE principles"""
    
    def __init__(self, kernel: CQEKernel):
        self.kernel = kernel
        self.data_sources: Dict[str, CQEDataSource] = {}
        self.format_handlers: Dict[str, Callable] = {}
        self.output_formatters: Dict[str, Callable] = {}
        self.stream_processors: Dict[str, Callable] = {}
        
        # Initialize format handlers
        self._initialize_format_handlers()
        self._initialize_output_formatters()
        self._initialize_stream_processors()
    
    def _initialize_format_handlers(self):
        """Initialize handlers for different data formats"""
        self.format_handlers = {
            'json': self._handle_json,
            'csv': self._handle_csv,
            'xml': self._handle_xml,
            'yaml': self._handle_yaml,
            'text': self._handle_text,
            'binary': self._handle_binary,
            'pickle': self._handle_pickle,
            'sql': self._handle_sql,
            'html': self._handle_html,
            'markdown': self._handle_markdown,
            'python': self._handle_python_code,
            'javascript': self._handle_javascript,
            'image': self._handle_image_metadata,
            'audio': self._handle_audio_metadata,
            'video': self._handle_video_metadata
        }
    
    def _initialize_output_formatters(self):
        """Initialize output formatters for different targets"""
        self.output_formatters = {
            'json': self._format_as_json,
            'csv': self._format_as_csv,
            'xml': self._format_as_xml,
            'yaml': self._format_as_yaml,
            'text': self._format_as_text,
            'html': self._format_as_html,
            'markdown': self._format_as_markdown,
            'python': self._format_as_python,
            'cqe_native': self._format_as_cqe_native
        }
    
    def _initialize_stream_processors(self):
        """Initialize stream processors for real-time data"""
        self.stream_processors = {
            'line_by_line': self._process_line_stream,
            'chunk_based': self._process_chunk_stream,
            'event_driven': self._process_event_stream,
            'continuous': self._process_continuous_stream
        }
    
    def register_data_source(self, source_type: str, location: str, 
                           format: str = None, encoding: str = 'utf-8',
                           metadata: Dict[str, Any] = None) -> str:
        """Register a new data source"""
        source_id = hashlib.md5(f"{source_type}:{location}".encode()).hexdigest()
        
        # Auto-detect format if not provided
        if format is None:
            format = self._detect_format(location, source_type)
        
        data_source = CQEDataSource(
            source_id=source_id,
            source_type=source_type,
            location=location,
            format=format,
            encoding=encoding,
            metadata=metadata or {}
        )
        
        self.data_sources[source_id] = data_source
        
        # Create source atom
        source_atom = CQEAtom(
            data={
                'source_id': source_id,
                'type': 'data_source',
                'source_type': source_type,
                'location': location,
                'format': format
            },
            metadata={'io_manager': True, 'data_source': True}
        )
        
        self.kernel.memory_manager.store_atom(source_atom)
        
        return source_id
    
    def ingest_data(self, source_id: str, chunk_size: int = 1000,
                   transform_function: Callable = None) -> List[str]:
        """Ingest data from source and convert to CQE atoms"""
        if source_id not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_id}")
        
        data_source = self.data_sources[source_id]
        atom_ids = []
        
        try:
            # Get data from source
            raw_data = self._fetch_data(data_source)
            
            # Process data using appropriate handler
            handler = self.format_handlers.get(data_source.format, self._handle_generic)
            processed_data = handler(raw_data, data_source)
            
            # Apply transformation if provided
            if transform_function:
                processed_data = transform_function(processed_data)
            
            # Convert to CQE atoms
            if isinstance(processed_data, list):
                # Handle list of items
                for i, item in enumerate(processed_data):
                    atom = CQEAtom(
                        data=item,
                        metadata={
                            'source_id': source_id,
                            'index': i,
                            'format': data_source.format,
                            'ingestion_timestamp': time.time()
                        }
                    )
                    atom_id = self.kernel.memory_manager.store_atom(atom)
                    atom_ids.append(atom_id)
            
            elif isinstance(processed_data, dict):
                # Handle dictionary
                for key, value in processed_data.items():
                    atom = CQEAtom(
                        data={'key': key, 'value': value},
                        metadata={
                            'source_id': source_id,
                            'key': key,
                            'format': data_source.format,
                            'ingestion_timestamp': time.time()
                        }
                    )
                    atom_id = self.kernel.memory_manager.store_atom(atom)
                    atom_ids.append(atom_id)
            
            else:
                # Handle single item
                atom = CQEAtom(
                    data=processed_data,
                    metadata={
                        'source_id': source_id,
                        'format': data_source.format,
                        'ingestion_timestamp': time.time()
                    }
                )
                atom_id = self.kernel.memory_manager.store_atom(atom)
                atom_ids.append(atom_id)
        
        except Exception as e:
            # Create error atom
            error_atom = CQEAtom(
                data={
                    'error': str(e),
                    'source_id': source_id,
                    'operation': 'ingest_data'
                },
                metadata={'error': True, 'source_id': source_id}
            )
            error_id = self.kernel.memory_manager.store_atom(error_atom)
            atom_ids.append(error_id)
        
        return atom_ids
    
    def export_data(self, atom_ids: List[str], output_format: str,
                   output_location: str, parameters: Dict[str, Any] = None) -> bool:
        """Export CQE atoms to external format"""
        if parameters is None:
            parameters = {}
        
        try:
            # Retrieve atoms
            atoms = []
            for atom_id in atom_ids:
                atom = self.kernel.memory_manager.retrieve_atom(atom_id)
                if atom:
                    atoms.append(atom)
            
            if not atoms:
                return False
            
            # Format data
            formatter = self.output_formatters.get(output_format, self._format_as_generic)
            formatted_data = formatter(atoms, parameters)
            
            # Write to output location
            self._write_data(formatted_data, output_location, output_format, parameters)
            
            return True
        
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def stream_process(self, source_id: str, processor_type: str,
                      callback: Callable[[List[CQEAtom]], None],
                      parameters: Dict[str, Any] = None) -> bool:
        """Process data stream in real-time"""
        if source_id not in self.data_sources:
            return False
        
        if processor_type not in self.stream_processors:
            return False
        
        data_source = self.data_sources[source_id]
        processor = self.stream_processors[processor_type]
        
        try:
            processor(data_source, callback, parameters or {})
            return True
        except Exception as e:
            print(f"Stream processing failed: {e}")
            return False
    
    def create_universal_adapter(self, data_sample: Any) -> Callable:
        """Create universal adapter for any data type"""
        def universal_adapter(data: Any) -> CQEAtom:
            # Analyze data structure
            data_type = type(data).__name__
            
            # Create appropriate CQE representation
            if isinstance(data, (str, int, float, bool)):
                # Primitive types
                return CQEAtom(
                    data=data,
                    metadata={'data_type': data_type, 'adapter': 'universal'}
                )
            
            elif isinstance(data, (list, tuple)):
                # Sequence types
                return CQEAtom(
                    data={
                        'type': 'sequence',
                        'length': len(data),
                        'items': data,
                        'item_types': [type(item).__name__ for item in data]
                    },
                    metadata={'data_type': data_type, 'adapter': 'universal'}
                )
            
            elif isinstance(data, dict):
                # Mapping types
                return CQEAtom(
                    data={
                        'type': 'mapping',
                        'keys': list(data.keys()),
                        'values': list(data.values()),
                        'size': len(data)
                    },
                    metadata={'data_type': data_type, 'adapter': 'universal'}
                )
            
            else:
                # Complex objects
                try:
                    # Try to serialize
                    serialized = json.dumps(data, default=str)
                    return CQEAtom(
                        data={
                            'type': 'complex_object',
                            'class': data_type,
                            'serialized': serialized,
                            'attributes': dir(data) if hasattr(data, '__dict__') else []
                        },
                        metadata={'data_type': data_type, 'adapter': 'universal'}
                    )
                except:
                    # Fallback to string representation
                    return CQEAtom(
                        data={
                            'type': 'unknown_object',
                            'class': data_type,
                            'string_repr': str(data)
                        },
                        metadata={'data_type': data_type, 'adapter': 'universal'}
                    )
        
        return universal_adapter
    
    # Format Handlers
    def _handle_json(self, data: str, source: CQEDataSource) -> Any:
        """Handle JSON data"""
        return json.loads(data)
    
    def _handle_csv(self, data: str, source: CQEDataSource) -> List[Dict[str, str]]:
        """Handle CSV data"""
        lines = data.strip().split('\n')
        reader = csv.DictReader(lines)
        return list(reader)
    
    def _handle_xml(self, data: str, source: CQEDataSource) -> Dict[str, Any]:
        """Handle XML data"""
        root = ET.fromstring(data)
        return self._xml_to_dict(root)
    
    def _handle_yaml(self, data: str, source: CQEDataSource) -> Any:
        """Handle YAML data"""
        return yaml.safe_load(data)
    
    def _handle_text(self, data: str, source: CQEDataSource) -> Dict[str, Any]:
        """Handle plain text data"""
        lines = data.split('\n')
        words = data.split()
        
        return {
            'content': data,
            'lines': lines,
            'line_count': len(lines),
            'words': words,
            'word_count': len(words),
            'character_count': len(data)
        }
    
    def _handle_binary(self, data: bytes, source: CQEDataSource) -> Dict[str, Any]:
        """Handle binary data"""
        return {
            'type': 'binary',
            'size': len(data),
            'base64': base64.b64encode(data).decode('ascii'),
            'hash': hashlib.md5(data).hexdigest()
        }
    
    def _handle_pickle(self, data: bytes, source: CQEDataSource) -> Any:
        """Handle pickled data"""
        return pickle.loads(data)
    
    def _handle_sql(self, data: str, source: CQEDataSource) -> List[Dict[str, Any]]:
        """Handle SQL query results"""
        # This would connect to database and execute query
        # For now, return parsed SQL structure
        return {
            'sql_query': data,
            'parsed': self._parse_sql(data)
        }
    
    def _handle_html(self, data: str, source: CQEDataSource) -> Dict[str, Any]:
        """Handle HTML data"""
        # Extract text content and structure
        text_content = re.sub(r'<[^>]+>', '', data)
        tags = re.findall(r'<([^>]+)>', data)
        
        return {
            'html': data,
            'text_content': text_content,
            'tags': tags,
            'tag_count': len(tags)
        }
    
    def _handle_markdown(self, data: str, source: CQEDataSource) -> Dict[str, Any]:
        """Handle Markdown data"""
        headers = re.findall(r'^#+\s+(.+)$', data, re.MULTILINE)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', data)
        
        return {
            'markdown': data,
            'headers': headers,
            'links': links,
            'header_count': len(headers),
            'link_count': len(links)
        }
    
    def _handle_python_code(self, data: str, source: CQEDataSource) -> Dict[str, Any]:
        """Handle Python code"""
        import ast
        
        try:
            tree = ast.parse(data)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
            
            return {
                'code': data,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'ast_valid': True
            }
        except SyntaxError:
            return {
                'code': data,
                'ast_valid': False,
                'syntax_error': True
            }
    
    def _handle_javascript(self, data: str, source: CQEDataSource) -> Dict[str, Any]:
        """Handle JavaScript code"""
        functions = re.findall(r'function\s+(\w+)', data)
        variables = re.findall(r'(?:var|let|const)\s+(\w+)', data)
        
        return {
            'code': data,
            'functions': functions,
            'variables': variables
        }
    
    def _handle_image_metadata(self, data: bytes, source: CQEDataSource) -> Dict[str, Any]:
        """Handle image metadata"""
        return {
            'type': 'image',
            'size': len(data),
            'format': source.metadata.get('image_format', 'unknown'),
            'hash': hashlib.md5(data).hexdigest()
        }
    
    def _handle_audio_metadata(self, data: bytes, source: CQEDataSource) -> Dict[str, Any]:
        """Handle audio metadata"""
        return {
            'type': 'audio',
            'size': len(data),
            'format': source.metadata.get('audio_format', 'unknown'),
            'hash': hashlib.md5(data).hexdigest()
        }
    
    def _handle_video_metadata(self, data: bytes, source: CQEDataSource) -> Dict[str, Any]:
        """Handle video metadata"""
        return {
            'type': 'video',
            'size': len(data),
            'format': source.metadata.get('video_format', 'unknown'),
            'hash': hashlib.md5(data).hexdigest()
        }
    
    def _handle_generic(self, data: Any, source: CQEDataSource) -> Dict[str, Any]:
        """Generic handler for unknown formats"""
        return {
            'type': 'generic',
            'format': source.format,
            'data': str(data),
            'size': len(str(data))
        }
    
    # Output Formatters
    def _format_as_json(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as JSON"""
        data = []
        for atom in atoms:
            data.append({
                'id': atom.id,
                'data': atom.data,
                'quad_encoding': atom.quad_encoding,
                'governance_state': atom.governance_state,
                'metadata': atom.metadata
            })
        
        return json.dumps(data, indent=parameters.get('indent', 2), default=str)
    
    def _format_as_csv(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as CSV"""
        if not atoms:
            return ""
        
        # Extract common fields
        fieldnames = ['id', 'data', 'governance_state']
        
        # Add metadata fields
        all_metadata_keys = set()
        for atom in atoms:
            all_metadata_keys.update(atom.metadata.keys())
        
        fieldnames.extend(sorted(all_metadata_keys))
        
        # Create CSV content
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for atom in atoms:
            row = {
                'id': atom.id,
                'data': str(atom.data),
                'governance_state': atom.governance_state
            }
            row.update(atom.metadata)
            writer.writerow(row)
        
        return output.getvalue()
    
    def _format_as_xml(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as XML"""
        root = ET.Element('cqe_atoms')
        
        for atom in atoms:
            atom_elem = ET.SubElement(root, 'atom')
            atom_elem.set('id', atom.id)
            atom_elem.set('governance_state', atom.governance_state)
            
            data_elem = ET.SubElement(atom_elem, 'data')
            data_elem.text = str(atom.data)
            
            metadata_elem = ET.SubElement(atom_elem, 'metadata')
            for key, value in atom.metadata.items():
                meta_elem = ET.SubElement(metadata_elem, key)
                meta_elem.text = str(value)
        
        return ET.tostring(root, encoding='unicode')
    
    def _format_as_yaml(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as YAML"""
        data = []
        for atom in atoms:
            data.append({
                'id': atom.id,
                'data': atom.data,
                'quad_encoding': list(atom.quad_encoding),
                'governance_state': atom.governance_state,
                'metadata': atom.metadata
            })
        
        return yaml.dump(data, default_flow_style=False)
    
    def _format_as_text(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as plain text"""
        lines = []
        for atom in atoms:
            lines.append(f"Atom ID: {atom.id}")
            lines.append(f"Data: {atom.data}")
            lines.append(f"Governance: {atom.governance_state}")
            lines.append(f"Quad: {atom.quad_encoding}")
            lines.append("---")
        
        return '\n'.join(lines)
    
    def _format_as_html(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as HTML"""
        html = ["<html><body><h1>CQE Atoms</h1>"]
        
        for atom in atoms:
            html.append(f"<div class='atom'>")
            html.append(f"<h3>Atom {atom.id}</h3>")
            html.append(f"<p><strong>Data:</strong> {atom.data}</p>")
            html.append(f"<p><strong>Governance:</strong> {atom.governance_state}</p>")
            html.append(f"<p><strong>Quad:</strong> {atom.quad_encoding}</p>")
            html.append("</div>")
        
        html.append("</body></html>")
        return '\n'.join(html)
    
    def _format_as_markdown(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as Markdown"""
        lines = ["# CQE Atoms\n"]
        
        for atom in atoms:
            lines.append(f"## Atom {atom.id}")
            lines.append(f"**Data:** {atom.data}")
            lines.append(f"**Governance:** {atom.governance_state}")
            lines.append(f"**Quad Encoding:** {atom.quad_encoding}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_as_python(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Format atoms as Python code"""
        lines = ["# CQE Atoms as Python data structures", ""]
        lines.append("atoms = [")
        
        for atom in atoms:
            lines.append("    {")
            lines.append(f"        'id': '{atom.id}',")
            lines.append(f"        'data': {repr(atom.data)},")
            lines.append(f"        'quad_encoding': {atom.quad_encoding},")
            lines.append(f"        'governance_state': '{atom.governance_state}',")
            lines.append(f"        'metadata': {atom.metadata}")
            lines.append("    },")
        
        lines.append("]")
        return '\n'.join(lines)
    
    def _format_as_cqe_native(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> bytes:
        """Format atoms in CQE native binary format"""
        # Serialize atoms using pickle for now
        # In practice, would use optimized CQE binary format
        return pickle.dumps([atom.to_dict() for atom in atoms])
    
    def _format_as_generic(self, atoms: List[CQEAtom], parameters: Dict[str, Any]) -> str:
        """Generic formatter"""
        return str([atom.to_dict() for atom in atoms])
    
    # Stream Processors
    def _process_line_stream(self, source: CQEDataSource, callback: Callable, parameters: Dict[str, Any]):
        """Process data line by line"""
        # Implementation for line-by-line processing
        pass
    
    def _process_chunk_stream(self, source: CQEDataSource, callback: Callable, parameters: Dict[str, Any]):
        """Process data in chunks"""
        # Implementation for chunk-based processing
        pass
    
    def _process_event_stream(self, source: CQEDataSource, callback: Callable, parameters: Dict[str, Any]):
        """Process event-driven data"""
        # Implementation for event-driven processing
        pass
    
    def _process_continuous_stream(self, source: CQEDataSource, callback: Callable, parameters: Dict[str, Any]):
        """Process continuous data stream"""
        # Implementation for continuous processing
        pass
    
    # Utility Methods
    def _detect_format(self, location: str, source_type: str) -> str:
        """Auto-detect data format"""
        if source_type == 'file':
            path = Path(location)
            extension = path.suffix.lower()
            
            format_map = {
                '.json': 'json',
                '.csv': 'csv',
                '.xml': 'xml',
                '.yaml': 'yaml', '.yml': 'yaml',
                '.txt': 'text',
                '.md': 'markdown',
                '.html': 'html', '.htm': 'html',
                '.py': 'python',
                '.js': 'javascript',
                '.pkl': 'pickle',
                '.sql': 'sql'
            }
            
            return format_map.get(extension, 'text')
        
        elif source_type == 'url':
            # Try to detect from URL or content-type
            return 'json'  # Default for URLs
        
        return 'generic'
    
    def _fetch_data(self, source: CQEDataSource) -> Union[str, bytes]:
        """Fetch data from source"""
        if source.source_type == 'file':
            path = Path(source.location)
            if source.format in ['binary', 'pickle', 'image', 'audio', 'video']:
                return path.read_bytes()
            else:
                return path.read_text(encoding=source.encoding)
        
        elif source.source_type == 'url':
            response = requests.get(source.location)
            if source.format in ['binary', 'pickle', 'image', 'audio', 'video']:
                return response.content
            else:
                return response.text
        
        elif source.source_type == 'database':
            # Database connection logic
            return self._fetch_from_database(source)
        
        elif source.source_type == 'stream':
            # Stream reading logic
            return self._fetch_from_stream(source)
        
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
    
    def _fetch_from_database(self, source: CQEDataSource) -> str:
        """Fetch data from database"""
        # Implementation for database fetching
        return ""
    
    def _fetch_from_stream(self, source: CQEDataSource) -> str:
        """Fetch data from stream"""
        # Implementation for stream fetching
        return ""
    
    def _write_data(self, data: Union[str, bytes], location: str, format: str, parameters: Dict[str, Any]):
        """Write data to output location"""
        path = Path(location)
        
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding=parameters.get('encoding', 'utf-8'))
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result['text'] = element.text.strip()
        
        # Add children
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def _parse_sql(self, sql: str) -> Dict[str, Any]:
        """Parse SQL query structure"""
        # Simple SQL parsing - in practice would use proper SQL parser
        sql_lower = sql.lower().strip()
        
        if sql_lower.startswith('select'):
            return {'type': 'select', 'query': sql}
        elif sql_lower.startswith('insert'):
            return {'type': 'insert', 'query': sql}
        elif sql_lower.startswith('update'):
            return {'type': 'update', 'query': sql}
        elif sql_lower.startswith('delete'):
            return {'type': 'delete', 'query': sql}
        else:
            return {'type': 'unknown', 'query': sql}

# Export main class
__all__ = ['CQEIOManager', 'CQEDataSource']
# cqe_kgram_tools.py
# Simple k-gram extraction to compare tokens vs snippets (shapes-first).

from collections import Counter
