import sys
sys.path.insert(0, ".")

from server.memory.structured import get_engine, init_db

if __name__ == "__main__":
    print("Initializing database...")
    engine = get_engine()
    init_db(engine)
    print("Done. Database created at data/memorylens.db")
