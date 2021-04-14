# Python class represent the entities
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

# Shared properties
class ProcessBase(BaseModel):
    status_code: Optional[int]
    status_name: Optional[str]
    description: Optional[str]
    document_id: Optional[int]
    create_date: Optional[datetime]
    update_date: Optional[datetime] = None
    class Config:
        orm_mode = True

# Model on DB
class ProcessInDB(ProcessBase):
    id: Optional[int]

# Create
class ProcessCreate(BaseModel):
    status_code: Optional[int]
    status_name: Optional[str]
    description: Optional[str]
    document_id: Optional[int]
    create_date: Optional[datetime]
        