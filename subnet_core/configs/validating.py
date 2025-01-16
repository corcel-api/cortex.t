from pydantic import BaseModel


class ValidatingConfig(BaseModel):
    synthetic_threshold: float = 0.1
    synthetic_batch_size: int = 4
    synthetic_concurrent_batches: int = 4
