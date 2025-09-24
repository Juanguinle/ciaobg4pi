# monitoring.py - System monitoring and health checks
#!/usr/bin/env python3
"""
System monitoring and health checks for the background removal service
"""

import psutil
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any
import asyncio
import aioredis
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    temperature: float
    gpu_usage: float
    queue_depth: int
    active_jobs: int
    uptime: float
    error_rate: float

class SystemMonitor:
    """System monitoring and alerting"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.logger = logging.getLogger(__name__)
        
    async def get_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.used / disk.total * 100
        
        # Temperature (Raspberry Pi specific)
        temp = self._get_cpu_temperature()
        
        # GPU usage (Hailo-8L specific)
        gpu_usage = await self._get_hailo_usage()
        
        # Queue metrics
        queue_depth, active_jobs = await self._get_queue_metrics()
        
        # Uptime
        uptime = time.time() - psutil.boot_time()
        
        # Error rate (from logs)
        error_rate = await self._calculate_error_rate()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk_percent,
            temperature=temp,
            gpu_usage=gpu_usage,
            queue_depth=queue_depth,
            active_jobs=active_jobs,
            uptime=uptime,
            error_rate=error_rate
        )
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature for Raspberry Pi"""
        try:
            temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_file.exists():
                temp = int(temp_file.read_text().strip()) / 1000.0
                return temp
            return 0.0
        except:
            return 0.0
    
    async def _get_hailo_usage(self) -> float:
        """Get Hailo-8L utilization (placeholder)"""
        try:
            # This would use actual Hailo SDK to get utilization
            # hailo_stats = hailo_sdk.get_device_stats()
            # return hailo_stats.utilization_percent
            return 0.0  # Placeholder
        except:
            return 0.0
    
    async def _get_queue_metrics(self) -> tuple[int, int]:
        """Get queue depth and active jobs from Redis"""
        try:
            redis = aioredis.from_url(self.redis_url)
            # This would query actual job queue
            queue_depth = 0  # Placeholder
            active_jobs = 0  # Placeholder
            await redis.close()
            return queue_depth, active_jobs
        except:
            return 0, 0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent operations"""
        try:
            # This would analyze recent job outcomes
            return 0.0  # Placeholder
        except:
            return 0.0
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        metrics = await self.get_system_metrics()
        
        health_status = {
            "status": "healthy",
            "metrics": metrics.__dict__,
            "alerts": []
        }
        
        # Check thresholds and generate alerts
        if metrics.cpu_percent > 80:
            health_status["alerts"].append({
                "level": "warning",
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%"
            })
        
        if metrics.memory_percent > 85:
            health_status["alerts"].append({
                "level": "warning", 
                "message": f"High memory usage: {metrics.memory_percent:.1f}%"
            })
        
        if metrics.temperature > 70:
            health_status["alerts"].append({
                "level": "critical",
                "message": f"High temperature: {metrics.temperature:.1f}°C"
            })
        
        if metrics.disk_usage_percent > 90:
            health_status["alerts"].append({
                "level": "critical",
                "message": f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            })
        
        if metrics.error_rate > 0.05:  # 5% error rate
            health_status["alerts"].append({
                "level": "warning",
                "message": f"High error rate: {metrics.error_rate:.1%}"
            })
        
        # Set overall status
        if any(alert["level"] == "critical" for alert in health_status["alerts"]):
            health_status["status"] = "critical"
        elif any(alert["level"] == "warning" for alert in health_status["alerts"]):
            health_status["status"] = "warning"
        
        return health_status

# Example usage and testing
if __name__ == "__main__":
    async def main():
        monitor = SystemMonitor()
        health = await monitor.check_system_health()
        print(json.dumps(health, indent=2))
    
    asyncio.run(main())