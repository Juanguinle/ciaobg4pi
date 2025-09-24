#!/usr/bin/env python3
"""
FTP Server and File System Watcher for Enterprise Background Removal Service

This module provides FTP/SFTP ingestion and directory monitoring capabilities
for automated background removal processing.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Set
import json
import time
import hashlib
from dataclasses import dataclass, asdict

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, TLS_FTPHandler
from pyftpdlib.servers import FTPServer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class ProcessingRequest:
    """Represents a file processing request from FTP or file watcher"""
    source_path: Path
    source_type: str  # 'ftp', 'watcher', 'webdav'
    client_info: Dict
    priority: int = 1
    created_at: float = None
    file_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class SecureFTPHandler(TLS_FTPHandler):
    """Secure FTP handler with custom processing logic"""
    
    def __init__(self, conn, server, ioloop=None):
        super().__init__(conn, server, ioloop)
        self.processing_queue = server.processing_queue
        self.logger = logging.getLogger(f"{__name__}.FTPHandler")
    
    def on_file_received(self, file_path):
        """Called when a file upload is completed"""
        try:
            file_path = Path(file_path)
            
            # Validate file type
            if not self._is_valid_image(file_path):
                self.logger.warning(f"Invalid file type uploaded: {file_path}")
                return
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            
            # Create processing request
            request = ProcessingRequest(
                source_path=file_path,
                source_type='ftp',
                client_info={
                    'username': self.username,
                    'remote_ip': self.remote_ip,
                    'upload_time': time.time()
                },
                file_hash=file_hash
            )
            
            # Add to processing queue
            self.processing_queue.add_request(request)
            self.logger.info(f"Queued FTP upload for processing: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling FTP upload {file_path}: {e}")
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image format"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return file_path.suffix.lower() in valid_extensions
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for deduplication"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class BackgroundRemovalFTPServer:
    """FTP Server for background removal service"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processing_queue = ProcessingQueue()
        self.server = None
        self.logger = logging.getLogger(f"{__name__}.FTPServer")
    
    def setup_server(self):
        """Configure and set up the FTP server"""
        # Create FTP root directory
        ftp_root = Path(self.config.get('ftp_root', './ftp'))
        ftp_root.mkdir(exist_ok=True)
        
        # Set up subdirectories
        (ftp_root / 'input').mkdir(exist_ok=True)
        (ftp_root / 'output').mkdir(exist_ok=True)
        (ftp_root / 'processed').mkdir(exist_ok=True)
        
        # Configure authorizer
        authorizer = DummyAuthorizer()
        
        # Add users from config
        users = self.config.get('ftp_users', [
            {'username': 'bgremoval', 'password': 'secure_password', 'home': str(ftp_root)}
        ])
        
        for user in users:
            authorizer.add_user(
                username=user['username'],
                password=user['password'],
                homedir=user['home'],
                perm='elradfmwMT'  # Full permissions
            )
        
        # Configure handler
        handler = SecureFTPHandler
        handler.authorizer = authorizer
        handler.banner = "Background Removal Service FTP Server"
        handler.max_cons = 256
        handler.max_cons_per_ip = 10
        
        # SSL/TLS configuration
        if self.config.get('ftp_ssl_enabled', False):
            handler.certfile = self.config.get('ftp_ssl_cert', './ssl/server.crt')
            handler.keyfile = self.config.get('ftp_ssl_key', './ssl/server.key')
        
        # Create server
        self.server = FTPServer(
            ('0.0.0.0', self.config.get('ftp_port', 21)),
            handler
        )
        self.server.processing_queue = self.processing_queue
        
        self.logger.info(f"FTP Server configured on port {self.config.get('ftp_port', 21)}")
    
    def start(self):
        """Start the FTP server"""
        if not self.server:
            self.setup_server()
        
        self.logger.info("Starting FTP server...")
        self.server.serve_forever()
    
    def stop(self):
        """Stop the FTP server"""
        if self.server:
            self.server.close_all()
            self.logger.info("FTP server stopped")

class FileSystemWatcher(FileSystemEventHandler):
    """Watch directory for new image files"""
    
    def __init__(self, watch_directory: Path, processing_queue):
        super().__init__()
        self.watch_directory = Path(watch_directory)
        self.processing_queue = processing_queue
        self.processed_files: Set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.FileWatcher")
        
        # Create watch directory if it doesn't exist
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # File processing delay to avoid partial files
        self.processing_delay = 2.0  # seconds
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        self._schedule_file_processing(Path(event.src_path))
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        self._schedule_file_processing(Path(event.src_path))
    
    def _schedule_file_processing(self, file_path: Path):
        """Schedule file for processing after delay"""
        if not self._is_valid_image(file_path):
            return
        
        # Use file hash to avoid duplicate processing
        try:
            file_hash = self._calculate_file_hash(file_path)
            
            if file_hash in self.processed_files:
                return
            
            self.processed_files.add(file_hash)
            
            # Schedule processing with delay
            asyncio.create_task(self._delayed_process_file(file_path, file_hash))
            
        except Exception as e:
            self.logger.error(f"Error scheduling file processing {file_path}: {e}")
    
    async def _delayed_process_file(self, file_path: Path, file_hash: str):
        """Process file after delay to ensure it's complete"""
        await asyncio.sleep(self.processing_delay)
        
        try:
            if not file_path.exists():
                self.processed_files.discard(file_hash)
                return
            
            # Create processing request
            request = ProcessingRequest(
                source_path=file_path,
                source_type='watcher',
                client_info={
                    'watch_directory': str(self.watch_directory),
                    'detected_at': time.time()
                },
                file_hash=file_hash
            )
            
            # Add to processing queue
            self.processing_queue.add_request(request)
            self.logger.info(f"Queued watched file for processing: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing watched file {file_path}: {e}")
            self.processed_files.discard(file_hash)
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image format"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return file_path.suffix.lower() in valid_extensions
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class ProcessingQueue:
    """Queue manager for processing requests from multiple sources"""
    
    def __init__(self):
        self.requests: List[ProcessingRequest] = []
        self.processing_callback = None
        self.logger = logging.getLogger(f"{__name__}.ProcessingQueue")
        self._lock = asyncio.Lock()
    
    def set_processing_callback(self, callback):
        """Set callback function for processing requests"""
        self.processing_callback = callback
    
    async def add_request(self, request: ProcessingRequest):
        """Add a processing request to the queue"""
        async with self._lock:
            # Check for duplicates based on file hash
            if request.file_hash:
                for existing in self.requests:
                    if existing.file_hash == request.file_hash:
                        self.logger.info(f"Duplicate file detected, skipping: {request.source_path}")
                        return
            
            # Insert based on priority (higher priority first)
            inserted = False
            for i, existing in enumerate(self.requests):
                if request.priority > existing.priority:
                    self.requests.insert(i, request)
                    inserted = True
                    break
            
            if not inserted:
                self.requests.append(request)
            
            self.logger.info(f"Added processing request: {request.source_path} (queue size: {len(self.requests)})")
            
            # Trigger processing if callback is set
            if self.processing_callback:
                asyncio.create_task(self._process_next())
    
    async def _process_next(self):
        """Process the next request in queue"""
        async with self._lock:
            if not self.requests:
                return
            
            request = self.requests.pop(0)
        
        try:
            if self.processing_callback:
                await self.processing_callback(request)
        except Exception as e:
            self.logger.error(f"Error processing request {request.source_path}: {e}")
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            'queue_length': len(self.requests),
            'requests': [
                {
                    'source_path': str(req.source_path),
                    'source_type': req.source_type,
                    'priority': req.priority,
                    'created_at': req.created_at,
                    'age_seconds': time.time() - req.created_at
                }
                for req in self.requests[:10]  # Show first 10
            ]
        }

class DirectoryWatcherService:
    """Service to manage directory watching"""
    
    def __init__(self, config: Dict, processing_queue: ProcessingQueue):
        self.config = config
        self.processing_queue = processing_queue
        self.observer = Observer()
        self.watchers: Dict[str, FileSystemWatcher] = {}
        self.logger = logging.getLogger(f"{__name__}.WatcherService")
    
    def add_watch_directory(self, directory: Path, name: str = None):
        """Add a directory to watch"""
        if name is None:
            name = str(directory)
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        watcher = FileSystemWatcher(directory, self.processing_queue)
        self.observer.schedule(watcher, str(directory), recursive=False)
        self.watchers[name] = watcher
        
        self.logger.info(f"Added watch directory: {directory}")
    
    def start(self):
        """Start directory watching"""
        # Add default watch directories from config
        watch_dirs = self.config.get('watch_directories', ['./watch_input'])
        
        for watch_dir in watch_dirs:
            self.add_watch_directory(Path(watch_dir))
        
        self.observer.start()
        self.logger.info("Directory watcher service started")
    
    def stop(self):
        """Stop directory watching"""
        self.observer.stop()
        self.observer.join()
        self.logger.info("Directory watcher service stopped")

# Integration with main background removal engine
class MultiSourceIngestionManager:
    """Manages multiple ingestion sources (FTP, file watcher, etc.)"""
    
    def __init__(self, background_removal_engine, config: Dict):
        self.engine = background_removal_engine
        self.config = config
        self.processing_queue = ProcessingQueue()
        self.ftp_server = None
        self.watcher_service = None
        self.logger = logging.getLogger(f"{__name__}.IngestionManager")
        
        # Set up processing callback
        self.processing_queue.set_processing_callback(self._process_request)
    
    async def _process_request(self, request: ProcessingRequest):
        """Process an ingestion request using the background removal engine"""
        try:
            self.logger.info(f"Processing {request.source_type} request: {request.source_path}")
            
            # Create a mock upload file object for the engine
            class MockUploadFile:
                def __init__(self, file_path: Path):
                    self.filename = file_path.name
                    self.content_type = f"image/{file_path.suffix[1:]}"
                    self._file_path = file_path
                
                async def read(self):
                    async with aiofiles.open(self._file_path, 'rb') as f:
                        return await f.read()
            
            # Create job using existing engine
            mock_file = MockUploadFile(request.source_path)
            job_id = self.engine.create_job(mock_file)
            
            # Copy source file to job input path
            job = self.engine.job_queue[job_id]
            shutil.copy2(request.source_path, job.input_path)
            
            # Process the job
            result_job = await self.engine.process_image(job_id)
            
            if result_job.status == "completed":
                # Handle successful processing based on source type
                await self._handle_successful_processing(request, result_job)
            else:
                self.logger.error(f"Processing failed for {request.source_path}: {result_job.error_message}")
                
        except Exception as e:
            self.logger.error(f"Error processing ingestion request {request.source_path}: {e}")
    
    async def _handle_successful_processing(self, request: ProcessingRequest, job):
        """Handle successful processing based on source type"""
        try:
            if request.source_type == 'ftp':
                # Move processed file to FTP output directory
                ftp_root = Path(self.config.get('ftp_root', './ftp'))
                output_dir = ftp_root / 'output'
                output_path = output_dir / f"{request.source_path.stem}_processed.png"
                shutil.copy2(job.output_path, output_path)
                
                # Move original to processed directory
                processed_dir = ftp_root / 'processed'
                processed_path = processed_dir / request.source_path.name
                shutil.move(request.source_path, processed_path)
                
            elif request.source_type == 'watcher':
                # Create output file next to original
                output_path = request.source_path.parent / f"{request.source_path.stem}_processed.png"
                shutil.copy2(job.output_path, output_path)
                
                # Optionally move original to processed subdirectory
                processed_dir = request.source_path.parent / 'processed'
                processed_dir.mkdir(exist_ok=True)
                processed_path = processed_dir / request.source_path.name
                shutil.move(request.source_path, processed_path)
            
            self.logger.info(f"Successfully processed {request.source_type} file: {request.source_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling successful processing for {request.source_path}: {e}")
    
    async def start_services(self):
        """Start all ingestion services"""
        try:
            # Start FTP server if enabled
            if self.config.get('ftp_enabled', True):
                self.ftp_server = BackgroundRemovalFTPServer(self.config)
                # Start FTP server in separate thread since it's blocking
                import threading
                ftp_thread = threading.Thread(target=self.ftp_server.start, daemon=True)
                ftp_thread.start()
                self.logger.info("FTP server started")
            
            # Start directory watcher if enabled
            if self.config.get('watcher_enabled', True):
                self.watcher_service = DirectoryWatcherService(self.config, self.processing_queue)
                self.watcher_service.start()
                self.logger.info("Directory watcher started")
            
            self.logger.info("All ingestion services started")
            
        except Exception as e:
            self.logger.error(f"Error starting ingestion services: {e}")
            raise
    
    def stop_services(self):
        """Stop all ingestion services"""
        try:
            if self.ftp_server:
                self.ftp_server.stop()
                
            if self.watcher_service:
                self.watcher_service.stop()
                
            self.logger.info("All ingestion services stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping ingestion services: {e}")
    
    def get_status(self) -> Dict:
        """Get status of all ingestion services"""
        return {
            'ftp_server': {
                'enabled': self.config.get('ftp_enabled', True),
                'running': self.ftp_server is not None
            },
            'directory_watcher': {
                'enabled': self.config.get('watcher_enabled', True),
                'running': self.watcher_service is not None,
                'watched_directories': list(self.watcher_service.watchers.keys()) if self.watcher_service else []
            },
            'processing_queue': self.processing_queue.get_queue_status()
        }

# Example configuration
DEFAULT_CONFIG = {
    'ftp_enabled': True,
    'ftp_port': 21,
    'ftp_ssl_enabled': False,
    'ftp_root': './ftp',
    'ftp_users': [
        {'username': 'bgremoval', 'password': 'secure_password_here', 'home': './ftp'}
    ],
    'watcher_enabled': True,
    'watch_directories': ['./watch_input', './drop_zone'],
    'webdav_enabled': False,
    'webdav_port': 8080
}