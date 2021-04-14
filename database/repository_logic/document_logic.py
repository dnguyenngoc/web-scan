from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from database.models import Document, Process
from sqlalchemy.orm import Session, joinedload, raiseload


# def read_by_type(db_session: Session, type: str) -> Document:
#     return db_session.query(Document).filter(Document.type == type) \
#                                      .options(joinedload('process')) \
#                                      .order_by(Document.import_date) \
#                                      .all()

