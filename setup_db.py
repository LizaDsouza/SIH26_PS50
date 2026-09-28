import sqlite3

# 1. Connect to (or create) the database file
conn = sqlite3.connect("known_drones.db")
cursor = conn.cursor()

# 2. Create a table for drone records
cursor.execute("""
    CREATE TABLE IF NOT EXISTS drones (
        drone_id TEXT PRIMARY KEY,
        owner TEXT,
        status TEXT
    )
""")

# 3. Add sample test data
sample_drones = [
    ("DRONE_001", "Friendly Patrol", "FRIEND"),
    ("DRONE_002", "Unknown Rogue Unit", "FOE")
]

cursor.executemany("""
    INSERT OR REPLACE INTO drones (drone_id, owner, status)
    VALUES (?, ?, ?)
""", sample_drones)

# 4. Save and close
conn.commit()
conn.close()

print("Database initialized successfully!")