from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from database.models import DocumentSplit


def create(db_session: Session, create) -> DocumentSplit:
    data = DocumentSplit(**create.dict())
    db_session.add(data)
    db_session.commit()
    db_session.refresh(data)
    return data


def read(db_session: Session, id: int) -> DocumentSplit:
    return db_session.query(DocumentSplit).filter(DocumentSplit.id == id).first()


def update(*, db_session: Session, id: int, update) -> DocumentSplit:
    update = db_session.query(DocumentSplit).filter(DocumentSplit.id == id).update(update, synchronize_session='evaluate')
    db_session.commit()
    return update


def delete(db_session: Session, id: int) -> DocumentSplit:
    query = db_session.query(DocumentSplit).filter(DocumentSplit.id == id).first()
    db_session.delete(query)
    db_session.commit()
    return query

