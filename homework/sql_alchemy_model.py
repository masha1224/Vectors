from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

# Tworzymy bazową klasę dla modeli SQLAlchemy
Base = declarative_base()

# Definiujemy klasę Img do przechowywania osadzeń obrazów
class Img(Base):
    __tablename__ = "images"
    __table_args__ = {'extend_existing': True}  # Aby rozszerzyć tabelę, jeśli już istnieje
    
    VECTOR_LENGTH: int = 512  
    
    # Kolumny w tabeli
    id: Mapped[int] = mapped_column(primary_key=True)  # Kolumna z identyfikatorem obrazu
    image_path: Mapped[str] = mapped_column(String)  # Kolumna z ścieżką do obrazu
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_LENGTH))  # Kolumna na osadzenie obrazu w postaci wektora

# Tworzymy połączenie z bazą danych PostgreSQL
db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="images",
)

engine = create_engine(db_url, echo=False)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

reset = False  # Set to True to drop and recreate the table

if reset:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

