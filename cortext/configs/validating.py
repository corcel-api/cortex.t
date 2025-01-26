from pydantic import BaseModel


class ValidatingConfig(BaseModel):
    synthetic_threshold: float
    synthetic_batch_size: int
    synthetic_concurrent_batches: int
