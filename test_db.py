from sqlalchemy import text
from app.db.session import SessionLocal

db = SessionLocal()
result = db.execute(text("SELECT NOW()")).fetchone()
print(result)
db.close()