import sqlite3

# Connect to SQLite database (create it if it doesn't exist)
conn = sqlite3.connect('db.sqlite3')

# Create a cursor object using the connection
cursor = conn.cursor()

# Create a table (e.g., documents) with columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL
    )
''')

# Insert sample data
cursor.execute('''
    INSERT INTO documents (title, content)
    VALUES
        ('Document 1', 'This is the content of document 1.'),
        ('Document 2', 'This is the content of document 2.'),
        ('Document 3', 'This is the content of document 3.')
''')

# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("SQLite database file 'db.sqlite3' created and populated successfully.")