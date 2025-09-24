#!/usr/bin/env python3
"""
Enterprise Background Removal Service - IMPROVED VERSION
Core implementation for Raspberry Pi 5 + Hailo-8L

- Implements realistic Hailo post-processing for YOLOv5-Seg.
- Uses Redis for persistent job storage, surviving restarts.
- Switched to async redis client for non-blocking operations.
- Centralized configuration.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid

import numpy as np
from PIL import Image
import cv2
import aiofiles
from redis import asyncio as redis # Use async redis client
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

try:
    import hailo_platform.pyhailort as hailort
    from hailo_platform.pyhailort import HEFFile, ConfigureParams, InputVStream, OutputVStream
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Data Models ---

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class ProcessingMethod(Enum):
    HAILO = "hailo"
    CPU_OPENCV = "cpu_opencv"

@dataclass
class ProcessingJob:
    job_id: str
    input_path: Path
    output_path: Path
    status: ProcessingStatus
    method: Optional[ProcessingMethod]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    error_message: Optional[str]
    progress: float
    retry_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataclass, converting None to empty strings for Redis."""
        data = asdict(self)
        # Convert specific types to string representations
        data['input_path'] = str(self.input_path)
        data['output_path'] = str(self.output_path)
        data['status'] = self.status.value
        data['method'] = self.method.value if self.method else "" # Handle optional Enum
        data['metadata'] = json.dumps(self.metadata) # Serialize metadata dict to JSON string

        # Convert any remaining None values to empty strings
        for key, value in data.items():
            if value is None:
                data[key] = ""
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingJob':
        """Deserialize dictionary from Redis, converting empty strings back to None."""
        # Convert empty strings back to None for optional fields
        for key in ['started_at', 'completed_at', 'error_message', 'method']:
            if key in data and data[key] == "":
                data[key] = None
        
        # Convert types back from strings
        data['input_path'] = Path(data['input_path'])
        data['output_path'] = Path(data['output_path'])
        data['status'] = ProcessingStatus(data['status'])
        if data.get('method'):
            data['method'] = ProcessingMethod(data['method'])
        
        # Deserialize metadata from JSON string
        if 'metadata' in data and data['metadata']:
            data['metadata'] = json.loads(data['metadata'])
        else:
            data['metadata'] = {}

        # Convert numeric types that might be strings from Redis
        for key in ['created_at', 'progress', 'retry_count', 'started_at', 'completed_at']:
             if key in data and data[key] is not None:
                try:
                    data[key] = float(data[key]) if '.' in str(data[key]) else int(data[key])
                except (ValueError, TypeError):
                    # Handle cases where conversion might fail for an empty string that became None
                    pass

        # Filter for fields that exist in the dataclass to avoid errors
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls(**filtered_data)

# --- Configuration ---

class ProcessingConfig:
    def __init__(self):
        self.max_resolution = (1920, 1080)
        self.max_retries = 3
        self.hailo_enabled = HAILO_AVAILABLE
        self.temp_dir = Path("/tmp/bg_removal")
        self.output_dir = Path("./data/outputs")
        self.cleanup_age_hours = 24
        self.hailo_model_path = Path("./models/yolov5n_seg.hef")
        self.hailo_confidence_threshold = 0.4 # Confidence for object detection

        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

# --- Core Processing Classes ---

class HailoBackgroundRemover:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = None
        self.network_group = None
        self.hef = None
        self.initialized = False

    async def initialize(self) -> bool:
        if not HAILO_AVAILABLE:
            logger.warning("Hailo SDK not available, cannot initialize.")
            return False
        if not self.config.hailo_model_path.exists():
            logger.error(f"Hailo model file not found: {self.config.hailo_model_path}")
            return False
        try:
            logger.info("Initializing Hailo-8L with YOLOv5n segmentation...")
            self.device = hailort.Device()
            self.hef = HEFFile(str(self.config.hailo_model_path))
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=hailort.HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(self.hef, configure_params)[0]
            self.initialized = True
            logger.info("Hailo-8L initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Hailo model: {e}")
            return False

    def _preprocess(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        original_shape = image.shape[:2] # H, W
        resized = cv2.resize(image, target_size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0), original_shape

    def _postprocess(self, outputs: list, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Post-process YOLOv5-Seg output to create a person mask.
        NOTE: This is a plausible implementation. The exact output tensor indices and shapes
        might need to be adjusted based on the specific model compilation.
        """
        # Assumed output format for YOLOv5-Seg:
        # outputs[0]: Detections [batch, num_anchors, 85 classes + 32 mask coeffs]
        # outputs[1]: Mask prototypes [batch, 32 mask coeffs, h/4, w/4]
        detections = outputs[0][0]
        mask_prototypes = outputs[1][0]
        
        # Filter for 'person' class (class_id 0 in COCO) with high confidence
        person_detections = detections[
            (detections[:, 4] > self.config.hailo_confidence_threshold) &
            (np.argmax(detections[:, 5:85], axis=1) == 0)
        ]
        
        if len(person_detections) == 0:
            logger.warning("No person detected by Hailo model.")
            return np.zeros(original_shape, dtype=np.uint8)

        # Get mask coefficients for detected persons
        mask_coeffs = person_detections[:, 85:]
        
        # Combine mask prototypes with coefficients (matrix multiplication)
        segmentation_masks = np.matmul(mask_coeffs, mask_prototypes.reshape(32, -1))
        segmentation_masks = segmentation_masks.reshape(-1, 160, 160) # Reshape to proto H, W

        # Apply sigmoid, threshold, and combine masks
        segmentation_masks = 1 / (1 + np.exp(-segmentation_masks)) # Sigmoid
        masks_binary = (segmentation_masks > 0.5).astype(np.uint8)
        
        # Combine all person masks into one
        final_mask = np.max(masks_binary, axis=0)
        
        # Resize mask to original image size
        resized_mask = cv2.resize(final_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized_mask * 255


    async def remove_background(self, image: np.ndarray) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Hailo model not initialized")
        
        start_time = time.time()
        input_data, original_shape = self._preprocess(image)

        with InputVStream.create(self.network_group) as input_vstream, \
             OutputVStream.create(self.network_group, self.hef.get_output_vstream_infos()[0]) as detections_vstream, \
             OutputVStream.create(self.network_group, self.hef.get_output_vstream_infos()[1]) as masks_vstream:
            
            input_vstream.send(input_data)
            detections = detections_vstream.recv()
            masks = masks_vstream.recv()

        person_mask = self._postprocess([detections, masks], original_shape)

        # Create RGBA image and apply mask to alpha channel
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        rgba_image[:, :, 3] = person_mask

        logger.info(f"Hailo processing completed in {time.time() - start_time:.2f}s")
        return rgba_image

class CPUBackgroundRemover:
    async def remove_background(self, image: np.ndarray) -> np.ndarray:
        start_time = time.time()
        height, width = image.shape[:2]
        rect = (width//10, height//10, 8*width//10, 8*height//10)
        mask = np.zeros((height, width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply morphological operations for cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        rgba_image[:, :, 3] = final_mask * 255
        
        logger.info(f"CPU (GrabCut) processing completed in {time.time() - start_time:.2f}s")
        return rgba_image

# --- Job Management ---

class JobStore:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.prefix = "bg_removal_job"

    async def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        job_key = f"{self.prefix}:{job_id}"
        job_data = await self.redis_client.hgetall(job_key)
        if not job_data:
            return None
        return ProcessingJob.from_dict(job_data)

    async def save_job(self, job: ProcessingJob):
        job_key = f"{self.prefix}:{job.job_id}"
        await self.redis_client.hset(job_key, mapping=job.to_dict())

    async def get_all_job_ids(self) -> list[str]:
        keys = await self.redis_client.keys(f"{self.prefix}:*")
        return [key.split(':')[-1] for key in keys]

# --- Main Engine ---

class BackgroundRemovalEngine:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.hailo_remover = HailoBackgroundRemover(config)
        self.cpu_remover = CPUBackgroundRemover()
        self.redis_client = redis.from_url("redis://localhost", decode_responses=True)
        self.job_store = JobStore(self.redis_client)

    async def initialize(self):
        try:
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.critical(f"Redis connection failed: {e}. The service cannot run without Redis.")
            raise
        
        if self.config.hailo_enabled:
            await self.hailo_remover.initialize()

    async def process_image_task(self, job_id: str):
        job = await self.job_store.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found in store for processing.")
            return

        try:
            job.status = ProcessingStatus.PROCESSING
            job.started_at = time.time()
            job.progress = 0.1
            await self.job_store.save_job(job)
            
            image = cv2.imread(str(job.input_path))
            if image is None: raise ValueError("Could not load image file.")
            
            job.progress = 0.2; await self.job_store.save_job(job)

            # Choose method and process
            if self.hailo_remover.initialized:
                job.method = ProcessingMethod.HAILO
                result_rgba = await self.hailo_remover.remove_background(image)
            else:
                job.method = ProcessingMethod.CPU_OPENCV
                result_rgba = await self.cpu_remover.remove_background(image)
            
            job.progress = 0.9; await self.job_store.save_job(job)
            
            # Save result
            output_image = Image.fromarray(result_rgba, 'RGBA')
            output_image.save(job.output_path, 'PNG')
            
            job.status = ProcessingStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = time.time()
            logger.info(f"Job {job_id} completed successfully with {job.method.value}.")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            
            # Retry logic
            if job.retry_count < self.config.max_retries:
                job.retry_count += 1
                job.status = ProcessingStatus.RETRYING
                await self.job_store.save_job(job)
                logger.info(f"Retrying job {job_id} (attempt {job.retry_count}) in {2**job.retry_count}s")
                await asyncio.sleep(2 ** job.retry_count)
                await self.process_image_task(job_id) # Recursive call for retry
                return

        await self.job_store.save_job(job)

    async def create_job(self, file: UploadFile) -> ProcessingJob:
        job_id = str(uuid.uuid4())
        input_path = self.config.temp_dir / f"{job_id}{Path(file.filename).suffix}"
        output_path = self.config.output_dir / f"{job_id}.png"
        
        job = ProcessingJob(
            job_id=job_id, input_path=input_path, output_path=output_path,
            status=ProcessingStatus.PENDING, method=None, created_at=time.time(),
            started_at=None, completed_at=None, error_message=None,
            progress=0.0, retry_count=0, metadata={}
        )
        
        async with aiofiles.open(input_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
            
        await self.job_store.save_job(job)
        return job

    async def cleanup_old_files(self):
        cutoff_time = time.time() - (self.config.cleanup_age_hours * 3600)
        job_ids = await self.job_store.get_all_job_ids()
        
        for job_id in job_ids:
            job = await self.job_store.get_job(job_id)
            if job and job.created_at < cutoff_time:
                job.input_path.unlink(missing_ok=True)
                job.output_path.unlink(missing_ok=True)
                await self.job_store.redis_client.delete(f"{self.job_store.prefix}:{job_id}")
                logger.info(f"Cleaned up old job {job_id}")

# --- FastAPI Application ---

app = FastAPI(title="Enterprise Background Removal Service", version="1.1.0")
config = ProcessingConfig()
engine = BackgroundRemovalEngine(config)

# --- API Models ---
class JobResponse(BaseModel):
    job_id: str
    status: str

class StatusResponse(BaseModel):
    job: Dict[str, Any]

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    await engine.initialize()
    app.state.cleanup_task = asyncio.create_task(periodic_cleanup())
    app.state.startup_time = time.time()

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600) # Run every hour
        logger.info("Running periodic cleanup task...")
        await engine.cleanup_old_files()

@app.post("/api/v1/process", response_model=JobResponse)
async def create_processing_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    job = await engine.create_job(file)
    background_tasks.add_task(engine.process_image_task, job.job_id)
    
    return JobResponse(job_id=job.job_id, status=job.status.value)

@app.get("/api/v1/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    job = await engine.job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return StatusResponse(job=job.to_dict())

@app.get("/api/v1/result/{job_id}")
async def get_job_result(job_id: str):
    job = await engine.job_store.get_job(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not complete. Current status: {job.status.value}")
    if not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk.")
    
    return FileResponse(job.output_path, media_type="image/png")

@app.get("/api/v1/health")
async def health_check():
    redis_ping = await engine.redis_client.ping()
    return {
        "status": "healthy",
        "hailo_initialized": engine.hailo_remover.initialized,
        "redis_connected": redis_ping,
        "uptime_seconds": time.time() - app.state.startup_time,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
