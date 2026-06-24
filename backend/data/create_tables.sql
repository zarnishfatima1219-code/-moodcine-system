-- MoodCine Database Schema
-- Muhammad Arslan Khan (24160374)
-- University of Hertfordshire

CREATE TABLE users (
    user_id      SERIAL PRIMARY KEY,
    username     VARCHAR(50) UNIQUE NOT NULL,
    email        VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login   TIMESTAMP
);

CREATE TABLE movies (
    movie_id     INTEGER PRIMARY KEY,
    title        VARCHAR(255) NOT NULL,
    tmdb_id      INTEGER,
    overview     TEXT,
    vote_average FLOAT,
    poster_url   VARCHAR(500),
    cast         VARCHAR(500),
    genres       VARCHAR(255),
    release_date VARCHAR(20)
);

CREATE TABLE ratings (
    rating_id  SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(user_id),
    movie_id   INTEGER REFERENCES movies(movie_id),
    rating     INTEGER CHECK (rating BETWEEN 1 AND 5),
    rated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE emotion_logs (
    emotion_id       SERIAL PRIMARY KEY,
    user_id          INTEGER REFERENCES users(user_id),
    mood_text        TEXT,
    nlp_emotion      VARCHAR(20),
    nlp_confidence   FLOAT,
    fer_emotion      VARCHAR(20),
    fer_confidence   FLOAT,
    fused_emotion    VARCHAR(20),
    fused_confidence FLOAT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE recommendations (
    rec_id      SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    emotion_id  INTEGER REFERENCES emotion_logs(emotion_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    rank        INTEGER CHECK (rank BETWEEN 1 AND 10),
    score       FLOAT,
    explanation TEXT,
    mode        VARCHAR(20) CHECK (mode IN ('baseline','nlp','fer','fusion')),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_feedback (
    feedback_id        SERIAL PRIMARY KEY,
    user_id            INTEGER REFERENCES users(user_id),
    rec_id             INTEGER REFERENCES recommendations(rec_id),
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),
    relevance_score    INTEGER CHECK (relevance_score BETWEEN 1 AND 5),
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_ratings_user    ON ratings(user_id);
CREATE INDEX idx_ratings_movie   ON ratings(movie_id);
CREATE INDEX idx_emotion_user    ON emotion_logs(user_id);
CREATE INDEX idx_rec_user        ON recommendations(user_id);
CREATE INDEX idx_rec_emotion     ON recommendations(emotion_id);
CREATE INDEX idx_rec_mode        ON recommendations(mode);

