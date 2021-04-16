from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from database.models import Document, Status
from sqlalchemy.orm import Session, joinedload, raiseload


