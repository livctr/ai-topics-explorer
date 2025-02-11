import psycopg2

def initialize_db():
    # Set your connection parameters
    conn = psycopg2.connect(
        dbname="your_db_name",
        user="your_username",
        password="your_password",
        host="your_host",  # e.g., "localhost"
        port="your_port"   # e.g., "5432"
    )
    cur = conn.cursor()

    # Read the schema.sql file
    with open('schema.sql', 'r') as f:
        sql_content = f.read()

    # Split the file into individual SQL statements.
    # This simple split assumes that semicolons only appear at the end of statements.
    statements = sql_content.split(';')

    for statement in statements:
        stmt = statement.strip()
        if stmt:
            try:
                cur.execute(stmt)
            except Exception as e:
                print("Error executing statement:")
                print(stmt)
                print("Error:", e)
    
    # Commit the changes and close the connection
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    initialize_db()
