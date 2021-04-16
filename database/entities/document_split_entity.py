# Python class represent the entities
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

# Shared properties
class DocumentSplitBase(BaseModel):
    name: Optional[str]
    value: Optional[str] = None
    is_extracted: Optional[str] = False
    type_id: Optional[int]
    url: Optional[str]
    document_id: Optional[int]
    create_date: Optional[datetime]
    update_date: Optional[datetime] = None
    class Config:
        orm_mode = True

# Data on db
class DocumentSplitInDB(DocumentSplitBase):
    id: Optional[int]

# Create DocumentSplit info
class DocumentSplitCreate(BaseModel):
    name: Optional[str]
    value: Optional[str] = None
    is_extracted: Optional[str] = False
    type_id: Optional[int]
    url: Optional[str]
    document_id: Optional[int]
    create_date: Optional[datetime]
