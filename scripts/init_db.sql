CREATE TABLE IF NOT EXISTS stores (
    store_id INTEGER PRIMARY KEY,
    store_type VARCHAR(10),
    assortment VARCHAR(10),
    competition_distance FLOAT,
    competition_open_since_month INTEGER,
    competition_open_since_year INTEGER,
    promo2 INTEGER,
    promo2_since_week INTEGER,
    promo2_since_year INTEGER,
    promo_interval VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    date DATE NOT NULL,
    day_of_week INTEGER,
    sales INTEGER,
    customers INTEGER,
    is_open INTEGER,
    promo INTEGER,
    state_holiday VARCHAR(5),
    school_holiday INTEGER
);

CREATE TABLE IF NOT EXISTS weather (
    id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    date DATE NOT NULL,
    temperature_mean FLOAT,
    temperature_max FLOAT,
    temperature_min FLOAT,
    precipitation FLOAT,
    wind_speed FLOAT,
    UNIQUE(store_id, date)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    forecast_date DATE NOT NULL,
    target_date DATE NOT NULL,
    horizon INTEGER,
    model_name VARCHAR(100),
    predicted_sales FLOAT,
    confidence_low FLOAT,
    confidence_high FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sales_store_date ON sales(store_id, date);
CREATE INDEX idx_weather_store_date ON weather(store_id, date);
CREATE INDEX idx_forecasts_store_target ON forecasts(store_id, target_date);
