from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Boolean, DECIMAL
from sqlalchemy.orm import relationship
from database.db import Base, engine, db_session

    
class Status(Base):
    __tablename__ = "status"
    id = Column(Integer, primary_key=True, index=True)
    status_code = Column(Integer)
    status_name = Column(String, nullable=True)
    description = Column(String)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)

    
class DocumentType(Base):
    __tablename__ = "document_type"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)
    
    
class Document(Base):
    __tablename__ = "document"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    export_date = Column(DateTime, nullable=True, default=None)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)
    type_id = Column(Integer,ForeignKey('document_type.id'), nullable=False)
    status_id = Column(Integer,ForeignKey('status.id'), nullable=False)
    
    type = relationship('DocumentType', lazy = 'noload', foreign_keys=[type_id])
    status = relationship('Status', lazy = 'noload', foreign_keys=[status_id])
    document_split = relationship("DocumentSplit", lazy='noload', back_populates="document")

    
class DocumentSplit(Base):
    __tablename__ = "document_split"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    value = Column(String, nullable=True, default=None)
    is_extracted = Column(Boolean, nullable = False, default = False)
    url = Column(String, nullable=False)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)
    type_id = Column(Integer,ForeignKey('document_type.id'), nullable=False)
    document_id = Column(Integer, ForeignKey("document.id"), nullable=False)
    
    type = relationship('DocumentType', lazy = 'noload', foreign_keys=[type_id])
    document = relationship('Document', lazy='noload', uselist=False, back_populates="document_split")

    
class DocumentTransForm(Base):
    __tablename__ = "document_transform"
    id = Column(Integer, primary_key=True, index=True)
   
    
Base.metadata.create_all(bind=engine)