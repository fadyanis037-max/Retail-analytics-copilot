import sqlite3
import os

db_path = 'data/northwind.sqlite'
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

con = sqlite3.connect(db_path)
cur = con.cursor()
cur.execute('CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;')
cur.execute('CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";')
cur.execute('CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;')
cur.execute('CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;')
con.commit()
con.close()
print("Views created successfully.")
