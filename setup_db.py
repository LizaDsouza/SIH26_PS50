import sqlite3

def init_db():
    # Connects to database file (creates it if it does not exist)
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()

    # 1. Create registered entity profiles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        status TEXT NOT NULL
    )
    """)

    # 2. Create live telemetry detection logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        bounding_box TEXT,
        location TEXT,
        alert_level TEXT
    )
    """)

    # Populate initial sample data into entities if empty
    cursor.execute("SELECT COUNT(*) FROM entities")
    if cursor.fetchone()[0] == 0:
        sample_entities = [
            ("DJI Mavic 3", "Drone", "Authorized"),
            ("Security Officer A", "Person", "Authorized"),
            ("Unidentified Quadcopter", "Drone", "Unauthorized")
        ]
        cursor.executemany("""
            INSERT INTO entities (name, category, status)
            VALUES (?, ?, ?)
        """, sample_entities)

    conn.commit()
    conn.close()
    print("Database schema successfully created in known_drones.db")

if __name__ == "__main__":
    init_db()