#!/usr/bin/env python3
"""
Integration and Testing Suite for Enterprise Background Removal Service

This module provides comprehensive testing, benchmarking, and integration
capabilities for the background removal service.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import concurrent.futures
import subprocess
import psutil
import pytest
import tempfile
from PIL import Image
import numpy as np
import cv2

# Import our main components
from main import BackgroundRemovalEngine, ProcessingConfig, app
from ftp_and_watcher import MultiSourceIngestionManager, DEFAULT_CONFIG
from monitoring import SystemMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

@dataclass
class PerformanceMetrics:
    processing_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    input_size_mb: float
    output_size_mb: float
    quality_score: Optional[float] = None

class ImageTestGenerator:
    """Generate test images for benchmarking"""
    
    @staticmethod
    def create_test_image(width: int, height: int, complexity: str = "simple") -> np.ndarray:
        """Create synthetic test images with different complexities"""
        
        if complexity == "simple":
            # Simple gradient background with basic shapes
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Gradient background
            for y in range(height):
                for x in range(width):
                    image[y, x] = [x % 256, y % 256, (x + y) % 256]
            
            # Add simple shapes (subject)
            center_x, center_y = width // 2, height // 2
            cv2.circle(image, (center_x, center_y), min(width, height) // 4, (255, 255, 255), -1)
            cv2.rectangle(image, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 0, 0), -1)
            
        elif complexity == "complex":
            # Complex scene with fine details
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Add noise and details
            noise = np.random.randint(-30, 30, (height, width, 3))
            image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
            
            # Add fine details (simulating hair/fur)
            for i in range(100):
                x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                x2, y2 = x1 + np.random.randint(-20, 20), y1 + np.random.randint(-20, 20)
                x2, y2 = np.clip([x2, y2], 0, [width-1, height-1])
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
        elif complexity == "portrait":
            # Portrait-like image with person-shaped subject
            image = np.full((height, width, 3), [120, 150, 180], dtype=np.uint8)  # Background
            
            # Person shape (simplified)
            center_x, center_y = width // 2, height // 2
            
            # Head
            cv2.ellipse(image, (center_x, center_y - height//4), (width//8, height//6), 0, 0, 360, (220, 180, 160), -1)
            
            # Body
            cv2.rectangle(image, (center_x - width//6, center_y - height//8), 
                         (center_x + width//6, center_y + height//3), (100, 100, 200), -1)
            
            # Arms
            cv2.rectangle(image, (center_x - width//4, center_y - height//8), 
                         (center_x - width//6, center_y + height//4), (220, 180, 160), -1)
            cv2.rectangle(image, (center_x + width//6, center_y - height//8), 
                         (center_x + width//4, center_y + height//4), (220, 180, 160), -1)
        
        return image
    
    @staticmethod
    def save_test_image(image: np.ndarray, path: Path, format: str = "PNG"):
        """Save test image to file"""
        pil_image = Image.fromarray(image, 'RGB')
        pil_image.save(path, format)

class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.config = ProcessingConfig()
        self.engine = BackgroundRemovalEngine(self.config)
    
    async def initialize(self):
        """Initialize benchmark environment"""
        await self.engine.initialize()
    
    async def run_single_image_benchmark(self, width: int, height: int, complexity: str) -> PerformanceMetrics:
        """Benchmark processing of a single image"""
        
        # Generate test image
        test_image = ImageTestGenerator.create_test_image(width, height, complexity)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            ImageTestGenerator.save_test_image(test_image, tmp_path)
        
        try:
            # Measure system resources before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            cpu_percent_start = psutil.cpu_percent()
            
            # Create mock upload file
            class MockUploadFile:
                def __init__(self, file_path: Path):
                    self.filename = file_path.name
                    self.content_type = "image/png"
                    self._file_path = file_path
                
                async def read(self):
                    with open(self._file_path, 'rb') as f:
                        return f.read()
            
            # Process image
            mock_file = MockUploadFile(tmp_path)
            job_id = self.engine.create_job(mock_file)
            job = self.engine.job_queue[job_id]
            
            # Copy file to input path
            import shutil
            shutil.copy2(tmp_path, job.input_path)
            
            # Process
            result_job = await self.engine.process_image(job_id)
            
            # Measure resources after
            end_time = time.time()
            processing_time = end_time - start_time
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            cpu_percent_end = psutil.cpu_percent()
            cpu_usage = (cpu_percent_start + cpu_percent_end) / 2
            
            # File sizes
            input_size = tmp_path.stat().st_size / 1024 / 1024  # MB
            output_size = 0
            if result_job.output_path.exists():
                output_size = result_job.output_path.stat().st_size / 1024 / 1024  # MB
            
            return PerformanceMetrics(
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                input_size_mb=input_size,
                output_size_mb=output_size,
                quality_score=result_job.metadata.get('confidence', 0.0)
            )
            
        finally:
            # Cleanup
            tmp_path.unlink(missing_ok=True)
            if 'job' in locals() and job.input_path.exists():
                job.input_path.unlink(missing_ok=True)
            if 'job' in locals() and job.output_path.exists():
                job.output_path.unlink(missing_ok=True)
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        
        test_cases = [
            (512, 512, "simple"),
            (1024, 1024, "simple"),
            (2048, 2048, "simple"),
            (4096, 4096, "simple"),
            (1024, 1024, "complex"),
            (2048, 2048, "complex"),
            (1024, 1024, "portrait"),
            (2048, 2048, "portrait"),
        ]
        
        results = {}
        
        logger.info("Starting comprehensive performance benchmark...")
        
        for width, height, complexity in test_cases:
            test_name = f"{width}x{height}_{complexity}"
            logger.info(f"Running benchmark: {test_name}")
            
            try:
                metrics = await self.run_single_image_benchmark(width, height, complexity)
                results[test_name] = asdict(metrics)
                
                logger.info(f"  Processing time: {metrics.processing_time:.2f}s")
                logger.info(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
                logger.info(f"  Quality score: {metrics.quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {test_name}: {e}")
                results[test_name] = {"error": str(e)}
        
        # Calculate summary statistics
        successful_results = [r for r in results.values() if "error" not in r]
        
        if successful_results:
            processing_times = [r["processing_time"] for r in successful_results]
            memory_usages = [r["memory_usage_mb"] for r in successful_results]
            quality_scores = [r["quality_score"] for r in successful_results if r["quality_score"]]
            
            summary = {
                "total_tests": len(test_cases),
                "successful_tests": len(successful_results),
                "processing_time_stats": {
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "stdev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0
                },
                "memory_usage_stats": {
                    "min": min(memory_usages),
                    "max": max(memory_usages),
                    "mean": statistics.mean(memory_usages)
                },
                "quality_score_stats": {
                    "min": min(quality_scores) if quality_scores else 0,
                    "max": max(quality_scores) if quality_scores else 0,
                    "mean": statistics.mean(quality_scores) if quality_scores else 0
                }
            }
            
            results["summary"] = summary
        
        return results

class LoadTester:
    """Load testing for concurrent processing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def test_concurrent_processing(self, num_concurrent: int, total_requests: int) -> Dict[str, Any]:
        """Test concurrent API requests"""
        
        async def single_request(session: aiohttp.ClientSession, request_id: int) -> Dict:
            """Process a single API request"""
            start_time = time.time()
            
            try:
                # Generate test image
                test_image = ImageTestGenerator.create_test_image(1024, 1024, "simple")
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    ImageTestGenerator.save_test_image(test_image, tmp_path)
                
                # Upload image
                with open(tmp_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename='test.png', content_type='image/png')
                    
                    async with session.post(f"{self.base_url}/api/v1/process", data=data) as response:
                        if response.status != 200:
                            return {
                                "request_id": request_id,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "duration": time.time() - start_time
                            }
                        
                        result = await response.json()
                        job_id = result["job_id"]
                
                # Poll for completion
                while True:
                    async with session.get(f"{self.base_url}/api/v1/status/{job_id}") as response:
                        if response.status != 200:
                            return {
                                "request_id": request_id,
                                "success": False,
                                "error": f"Status check failed: HTTP {response.status}",
                                "duration": time.time() - start_time
                            }
                        
                        status = await response.json()
                        
                        if status["status"] == "completed":
                            return {
                                "request_id": request_id,
                                "success": True,
                                "duration": time.time() - start_time,
                                "processing_time": status.get("completed_at", 0) - status.get("started_at", 0)
                            }
                        elif status["status"] == "failed":
                            return {
                                "request_id": request_id,
                                "success": False,
                                "error": status.get("error_message", "Unknown error"),
                                "duration": time.time() - start_time
                            }
                        
                        await asyncio.sleep(0.5)  # Poll every 500ms
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - start_time
                }
            finally:
                # Cleanup
                if 'tmp_path' in locals():
                    tmp_path.unlink(missing_ok=True)
        
        # Run concurrent requests
        logger.info(f"Starting load test: {num_concurrent} concurrent, {total_requests} total")
        
        start_time = time.time()
        results = []
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(num_concurrent)
            
            async def limited_request(request_id):
                async with semaphore:
                    return await single_request(session, request_id)
            
            tasks = [limited_request(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks)
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        durations = [r["duration"] for r in successful_requests]
        processing_times = [r["processing_time"] for r in successful_requests if "processing_time" in r]
        
        analysis = {
            "total_requests": total_requests,
            "concurrent_requests": num_concurrent,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / total_requests * 100,
            "total_duration": total_duration,
            "throughput_rps": total_requests / total_duration,
            "response_time_stats": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "mean": statistics.mean(durations) if durations else 0,
                "median": statistics.median(durations) if durations else 0,
                "p95": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            },
            "processing_time_stats": {
                "min": min(processing_times) if processing_times else 0,
                "max": max(processing_times) if processing_times else 0,
                "mean": statistics.mean(processing_times) if processing_times else 0
            },
            "errors": [r["error"] for r in failed_requests][:10]  # Show first 10 errors
        }
        
        return analysis

class IntegrationTester:
    """Integration testing for all components"""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
    
    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all API endpoints"""
        results = []
        base_url = "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            start_time = time.time()
            try:
                async with session.get(f"{base_url}/api/v1/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results.append(TestResult(
                            test_name="health_endpoint",
                            success=True,
                            duration=time.time() - start_time,
                            metrics=health_data
                        ))
                    else:
                        results.append(TestResult(
                            test_name="health_endpoint",
                            success=False,
                            duration=time.time() - start_time,
                            error_message=f"HTTP {response.status}"
                        ))
            except Exception as e:
                results.append(TestResult(
                    test_name="health_endpoint",
                    success=False,
                    duration=time.time() - start_time,
                    error_message=str(e)
                ))
            
            # Test image processing endpoint
            start_time = time.time()
            try:
                # Create test image
                test_image = ImageTestGenerator.create_test_image(512, 512, "simple")
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    ImageTestGenerator.save_test_image(test_image, tmp_path)
                
                # Upload and process
                with open(tmp_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename='test.png', content_type='image/png')
                    
                    async with session.post(f"{base_url}/api/v1/process", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            job_id = result["job_id"]
                            
                            # Wait for completion
                            max_wait = 120  # 2 minutes
                            waited = 0
                            while waited < max_wait:
                                async with session.get(f"{base_url}/api/v1/status/{job_id}") as status_response:
                                    if status_response.status == 200:
                                        status = await status_response.json()
                                        if status["status"] == "completed":
                                            results.append(TestResult(
                                                test_name="image_processing_endpoint",
                                                success=True,
                                                duration=time.time() - start_time,
                                                metrics=status
                                            ))
                                            break
                                        elif status["status"] == "failed":
                                            results.append(TestResult(
                                                test_name="image_processing_endpoint",
                                                success=False,
                                                duration=time.time() - start_time,
                                                error_message=status.get("error_message", "Processing failed")
                                            ))
                                            break
                                
                                await asyncio.sleep(1)
                                waited += 1
                            else:
                                results.append(TestResult(
                                    test_name="image_processing_endpoint",
                                    success=False,
                                    duration=time.time() - start_time,
                                    error_message="Timeout waiting for processing"
                                ))
                        else:
                            results.append(TestResult(
                                test_name="image_processing_endpoint",
                                success=False,
                                duration=time.time() - start_time,
                                error_message=f"HTTP {response.status}"
                            ))
                
                # Cleanup
                tmp_path.unlink(missing_ok=True)
                
            except Exception as e:
                results.append(TestResult(
                    test_name="image_processing_endpoint",
                    success=False,
                    duration=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    async def test_system_components(self) -> List[TestResult]:
        """Test individual system components"""
        results = []
        
        # Test system monitor
        start_time = time.time()
        try:
            monitor = SystemMonitor()
            health = await monitor.check_system_health()
            
            results.append(TestResult(
                test_name="system_monitor",
                success=True,
                duration=time.time() - start_time,
                metrics=health
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="system_monitor",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            ))
        
        # Test background removal engine
        start_time = time.time()
        try:
            config = ProcessingConfig()
            engine = BackgroundRemovalEngine(config)
            await engine.initialize()
            
            results.append(TestResult(
                test_name="background_removal_engine",
                success=True,
                duration=time.time() - start_time,
                metrics={"hailo_available": engine.hailo_remover.initialized}
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="background_removal_engine",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            ))
        
        return results
    
    async def run_full_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("Starting full integration test suite...")
        
        all_results = []
        
        # Test API endpoints
        api_results = await self.test_api_endpoints()
        all_results.extend(api_results)
        
        # Test system components
        component_results = await self.test_system_components()
        all_results.extend(component_results)
        
        # Calculate summary
        successful_tests = [r for r in all_results if r.success]
        failed_tests = [r for r in all_results if not r.success]
        
        summary = {
            "total_tests": len(all_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(all_results) * 100,
            "total_duration": sum(r.duration for r in all_results),
            "test_results": [asdict(r) for r in all_results]
        }
        
        logger.info(f"Integration tests completed: {len(successful_tests)}/{len(all_results)} passed")
        
        return summary

# Main testing orchestrator
class TestSuite:
    """Main test suite orchestrator"""
    
    def __init__(self):
        self.performance_benchmark = PerformanceBenchmark()
        self.load_tester = LoadTester()
        self.integration_tester = IntegrationTester()
    
    async def run_all_tests(self, include_load_test: bool = False) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("Starting comprehensive test suite...")
        
        results = {}
        
        try:
            # Initialize performance benchmark
            await self.performance_benchmark.initialize()
            
            # Run integration tests
            logger.info("Running integration tests...")
            results["integration_tests"] = await self.integration_tester.run_full_integration_test()
            
            # Run performance benchmarks
            logger.info("Running performance benchmarks...")
            results["performance_benchmarks"] = await self.performance_benchmark.run_comprehensive_benchmark()
            
            # Run load tests (optional)
            if include_load_test:
                logger.info("Running load tests...")
                load_test_configs = [
                    {"concurrent": 5, "total": 25},
                    {"concurrent": 10, "total": 50},
                ]
                
                results["load_tests"] = {}
                for config in load_test_configs:
                    test_name = f"c{config['concurrent']}_t{config['total']}"
                    results["load_tests"][test_name] = await self.load_tester.test_concurrent_processing(
                        config["concurrent"], config["total"]
                    )
            
            # Generate overall summary
            results["overall_summary"] = self._generate_overall_summary(results)
            
            logger.info("Test suite completed successfully")
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary"""
        summary = {
            "timestamp": time.time(),
            "components_tested": list(results.keys()),
            "overall_status": "passed"
        }
        
        # Check integration test results
        if "integration_tests" in results:
            integration = results["integration_tests"]
            if integration["failed_tests"] > 0:
                summary["overall_status"] = "failed"
            summary["integration_success_rate"] = integration["success_rate"]
        
        # Check performance benchmarks
        if "performance_benchmarks" in results:
            perf = results["performance_benchmarks"]
            if "summary" in perf:
                perf_summary = perf["summary"]
                if perf_summary["successful_tests"] < perf_summary["total_tests"]:
                    summary["overall_status"] = "degraded"
                
                # Check if any processing times exceed target
                if perf_summary["processing_time_stats"]["max"] > 60:
                    summary["performance_warning"] = "Some tests exceeded 60s target"
        
        # Check load test results
        if "load_tests" in results:
            for test_name, load_result in results["load_tests"].items():
                if load_result["success_rate"] < 95:
                    summary["overall_status"] = "failed"
                    summary["load_test_warning"] = f"{test_name} had {load_result['success_rate']:.1f}% success rate"
        
        return summary

# CLI interface for testing
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Background Removal Service Test Suite")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance benchmarks only")
    parser.add_argument("--load", action="store_true", help="Run load tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests including load tests")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    suite = TestSuite()
    results = {}
    
    if args.integration or (not any([args.performance, args.load, args.all])):
        results = await suite.integration_tester.run_full_integration_test()
    elif args.performance:
        await suite.performance_benchmark.initialize()
        results = await suite.performance_benchmark.run_comprehensive_benchmark()
    elif args.load:
        results = await suite.load_tester.test_concurrent_processing(5, 25)
    elif args.all:
        results = await suite.run_all_tests(include_load_test=True)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
