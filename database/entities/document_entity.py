# Python class represent the entities
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

# Shared properties
class DocumentBase(BaseModel):
    name: Optional[str]
    type: Optional[str]
    url: Optional[str]
    create_date: Optional[datetime]
    export_date: Optional[datetime] = None
    class Config:
        orm_mode = True

class DocumentInDB(DocumentBase):
    id: Optional[int]

# Create Document info
class DocumentCreate(BaseModel):
    name: Optional[str]
    type: Optional[str]
    url: Optional[str]
    create_date: Optional[datetime]
        