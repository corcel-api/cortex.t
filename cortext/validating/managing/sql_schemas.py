from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from loguru import logger
from ...global_config import CONFIG

Base = declarative_base()


class MinerMetadata(Base):
    __tablename__ = "miner_metadata"

    uid = Column(Integer, primary_key=True)
    accumulate_score = Column(Float, default=0.0)
    credit = Column(Integer, default=CONFIG.bandwidth.min_credit)

    def __init__(self, uid, accumulate_score=0.0, credit=CONFIG.bandwidth.min_credit):
        self.uid = uid
        self.accumulate_score = accumulate_score
        self.credit = max(
            min(credit, CONFIG.bandwidth.max_credit), CONFIG.bandwidth.min_credit
        )

    def set_credit(self, credit: int):
        correct_credit = credit if credit >= CONFIG.bandwidth.min_credit else 0
        logger.info(f"{self.uid}: {self.credit} -> {correct_credit}")
        self.credit = correct_credit

    def to_dict(self):
        """Convert metadata to dictionary format."""
        return {
            "uid": self.uid,
            "accumulate_score": self.accumulate_score,
            "credit": self.credit,
        }
