import json

import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy import create_engine, text

DB_PATH = "datasoup/sql_generator/db/db.sqlite"


def populate_db():
    fake = Faker()

    # Generate users data
    num_users = 1000
    users = pd.DataFrame(
        {
            "user_id": range(1, num_users + 1),
            "first_name": [fake.first_name() for _ in range(num_users)],
            "last_name": [fake.last_name() for _ in range(num_users)],
            "email": [fake.email() for _ in range(num_users)],
        }
    )
    users = users.set_index("user_id")

    # Generate stores data
    num_stores = 50
    stores = pd.DataFrame(
        {
            "store_id": range(1, num_stores + 1),
            "store_name": [fake.company() for _ in range(num_stores)],
            "address": [fake.address() for _ in range(num_stores)],
            "city": [fake.city() for _ in range(num_stores)],
            "state": [fake.state() for _ in range(num_stores)],
            "zip_code": [fake.zipcode() for _ in range(num_stores)],
            "square_footage": [fake.random_int(min=5000, max=50000) for _ in range(num_stores)],
            "num_employees": [fake.random_int(min=20, max=200) for _ in range(num_stores)],
        }
    )
    stores = stores.set_index("store_id")

    # Generate items data
    categories = ["produce", "dairy", "bakery", "meat", "frozen", "snacks", "beverages"]
    num_items = 1000
    items = pd.DataFrame(
        {
            "item_id": range(1, num_items + 1),
            "item_name": [fake.text(max_nb_chars=20) for _ in range(num_items)],
            "category": [np.random.choice(categories) for _ in range(num_items)],
            "price": np.array([fake.pydecimal(left_digits=2, right_digits=2, positive=True) for _ in range(num_items)]).astype(float),
        }
    )
    items = items.set_index("item_id")

    # Generate transactions data
    num_transactions = 100000
    user_ids = np.random.randint(1, num_users + 1, size=num_transactions)
    store_ids = np.random.randint(1, num_stores + 1, size=num_transactions)
    transaction_dates = np.array([fake.date_time_between(start_date="-1y", end_date="now") for _ in range(num_transactions)])
    num_items_per_transaction = np.random.randint(1, 11, size=num_transactions)
    item_ids = np.random.randint(1, num_items + 1, size=np.sum(num_items_per_transaction))
    quantities = np.random.randint(1, 6, size=np.sum(num_items_per_transaction))
    prices = items.loc[item_ids, "price"].values

    transaction_ids = np.repeat(np.arange(num_transactions), num_items_per_transaction)
    user_ids_repeated = np.repeat(user_ids, num_items_per_transaction)
    store_ids_repeated = np.repeat(store_ids, num_items_per_transaction)
    transaction_dates_repeated = np.repeat(transaction_dates, num_items_per_transaction)

    transactions = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "item_id": item_ids,
            "quantity": quantities,
            "price": prices,
            "user_id": user_ids_repeated,
            "store_id": store_ids_repeated,
            "transaction_date": transaction_dates_repeated,
        }
    )

    transactions = transactions.reset_index().rename(columns={"index": "item_scan_id"})

    sql_file = "datasoup/sql_generator/db/db.sql"
    with open(sql_file) as f:
        table_create_statements = f.read().split(";")
        table_create_statements = [stmt.strip() for stmt in table_create_statements if stmt.strip()]
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.connect() as conn:
        for statement in table_create_statements:
            conn.execute(text(statement))
        num_users_inserted = users.to_sql("users", conn, if_exists="replace", index=False)
        assert num_users_inserted == num_users
        num_stores_inserted = stores.to_sql("stores", conn, if_exists="replace", index=False)
        assert num_stores_inserted == num_stores
        num_items_inserted = items.to_sql("items", conn, if_exists="replace", index=False)
        assert num_items_inserted == num_items
        num_transactions_inserted = transactions.to_sql("transactions", conn, if_exists="replace", index=False)
        assert num_transactions_inserted == len(transactions)
        conn.commit()


def execute_query(query: str):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.fetchall()


def get_table_metadata():
    with open("datasoup/sql_generator/db/db_metadata.json") as f:
        tables = json.load(f)
    return tables


if __name__ == "__main__":
    # populate_db()
    print(execute_query("SELECT * FROM users"))
