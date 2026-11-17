from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    nim = Column(String(20), unique=True, nullable=False, index=True)
    nama = Column(String(100), nullable=False)
    prodi = Column(String(100), nullable=False)
    kelas = Column(String(50), nullable=False)
    
    attendances = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    confidence_score = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="attendances")