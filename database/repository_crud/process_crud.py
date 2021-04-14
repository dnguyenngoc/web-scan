from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from database.models import Process


def create(db_session: Session, create) -> Process:
    data = Process(**create.dict())
    db_session.add(data)
    db_session.commit()
    db_session.refresh(data)
    return data


def read(db_session: Session, id: int) -> Process:
    return db_session.query(Process).filter(Process.id == id).first()


def update(*, db_session: Session, id: int, update) -> Process:
    update = db_session.query(Process).filter(Process.id == id).update(update, synchronize_session='evaluate')
    db_session.commit()
    return update


def delete(db_session: Session, id: int) -> Process:
    query = db_session.query(Process).filter(Process.id == id).first()
    db_session.delete(query)
    db_session.commit()
    return query

