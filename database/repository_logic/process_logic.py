from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from database.models import Document, Process
from sqlalchemy.orm import Session, joinedload, raiseload


def read_by_status_name_join_load(db_session: Session, status_name: str) -> Process:
    return db_session.query(Process) \
                     .options(joinedload('document')) \
                     .filter(Process.status_name == status_name).all()

def read_by_status_name_and_type_join_load(db_session: Session, status_name: str, type: str) -> Process:
    return db_session.query(Process) \
                     .options(joinedload('document')) \
                     .filter(Process.status_name == status_name, Document.type == type).all()


