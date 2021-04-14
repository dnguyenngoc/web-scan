from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Boolean, DECIMAL
from sqlalchemy.orm import relationship
from database.db import Base, engine


class Document(Base):
    __tablename__ = "document"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    url = Column(String, nullable=False)
    export_date = Column(DateTime, nullable=True, default=None)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)
    process = relationship("Process", lazy='noload', uselist=False, back_populates="document")
    
class Process(Base):
    __tablename__ = "process"
    id = Column(Integer, primary_key=True, index=True)
    status_code = Column(Integer)
    status_name = Column(String, nullable=True)
    description = Column(String)
    document_id = Column(Integer, ForeignKey("document.id"), nullable=True)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)
    document = relationship('Document', lazy='noload', uselist=False, back_populates="process")
    

Base.metadata.create_all(bind=engine)