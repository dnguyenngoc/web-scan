from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session, joinedload, raiseload
from databases.models import DocumentField
from datetime import datetime
from sqlalchemy import distinct
from sqlalchemy import func
