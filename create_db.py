import sqlite3

conn = sqlite3.connect('database.db')
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    name TEXT,
    email TEXT,
    mobile TEXT,
    password TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")