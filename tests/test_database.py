"""
MoodCine — Unit Tests: Database Connection
==========================================
Run: python -m pytest tests/test_database.py -v
"""
import psycopg2
import pytest

DB_CONFIG = {
    "host": "localhost",
    "database": "moodcine",
    "user": "postgres",
    "password": "moodcine123",
    "port": 5432
}

EXPECTED_TABLES = [
    "users", "movies", "ratings",
    "emotion_logs", "recommendations", "user_feedback"
]

@pytest.fixture(scope="module")
def db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    yield conn
    conn.close()

def test_database_connection(db_connection):
    """Test PostgreSQL connection works"""
    assert db_connection is not None
    assert not db_connection.closed

def test_all_tables_exist(db_connection):
    """Test all 6 required tables exist"""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cursor.fetchall()]
    for table in EXPECTED_TABLES:
        assert table in tables, f"Table '{table}' missing!"

def test_users_table_columns(db_connection):
    """Test users table has correct columns"""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'users'
    """)
    cols = [r[0] for r in cursor.fetchall()]
    for col in ["user_id","username","email","password_hash"]:
        assert col in cols

def test_recommendations_has_mode_column(db_connection):
    """Test recommendations table has mode column for research"""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'recommendations'
    """)
    cols = [r[0] for r in cursor.fetchall()]
    assert "mode" in cols, "mode column needed for research comparison"

def test_movies_table_columns(db_connection):
    """Test movies table has TMDB enrichment columns"""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'movies'
    """)
    cols = [r[0] for r in cursor.fetchall()]
    for col in ["movie_id","title","tmdb_id","poster_url","overview"]:
        assert col in cols
