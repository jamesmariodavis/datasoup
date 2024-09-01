-- Create the tables
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE,
    first_shopped_at TIMESTAMP
);

DROP TABLE IF EXISTS stores;
CREATE TABLE stores (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100),
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    square_footage INTEGER,
    num_employees INTEGER
);

DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    store_id INTEGER,
    transaction_date TIMESTAMP,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);
