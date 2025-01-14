from pydantic import BaseModel, Field


class TaskSchedulerConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    max_synthetic_queue_size: int = Field(default=1000)
    refill_threshold: int = Field(default=500)
    refill_target: int = Field(default=1000)
