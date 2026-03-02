import sqlite3

DB_NAME = 'chatbot_data.db'

def setup_database():
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()
    
    # Table for the Intent Tags and their Responses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT UNIQUE,
            responses TEXT  -- We'll store responses as a pipe-separated string
        )
    ''')
    
    # Table for the training patterns linked to an intent
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id INTEGER,
            pattern_text TEXT,
            FOREIGN KEY (intent_id) REFERENCES intents (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def menu():
    while True:
        print("\n--- CHATBOT KNOWLEDGE MANAGER ---")
        print("1. Create Database")
        print("2. View All Intents")
        print("3. Add New Intent (Tag, Patterns, Responses)")
        print("4. Delete an Intent")
        print("5. Exit")
        
        choice = input("\nSelect an option: ")
        
        if choice == '1':
            setup_database()
        elif choice == '2':
            view_intents()
        elif choice == '3':
            add_intent_flow()
        elif choice == '4':
            delete_intent_flow()
        elif choice == '5':
            break
        else:
            print("Invalid choice, try again.")

def view_intents():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, tag, responses FROM intents")
    rows = cursor.fetchall()
    
    if not rows:
        print("\n[!] The database is empty.")
    else:
        for row in rows:
            print(f"\nID: {row[0]} | TAG: {row[1]}")
            # Get patterns for this intent
            cursor.execute("SELECT pattern_text FROM patterns WHERE intent_id=?", (row[0],))
            patterns = [p[0] for p in cursor.fetchall()]
            print(f"Patterns: {', '.join(patterns)}")
            print(f"Responses: {row[2].replace('|', ' / ')}")
    conn.close()

def add_intent_flow():
    tag = input("Enter intent tag (e.g., 'greeting'): ").strip().lower()
    patterns_raw = input("Enter patterns separated by commas: ")
    responses_raw = input("Enter responses separated by commas: ")
    
    patterns = [p.strip() for p in patterns_raw.split(',')]
    responses = [r.strip() for r in responses_raw.split(',')]
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO intents (tag, responses) VALUES (?, ?)", (tag, "|".join(responses)))
        intent_id = cursor.lastrowid
        for p in patterns:
            cursor.execute("INSERT INTO patterns (intent_id, pattern_text) VALUES (?, ?)", (intent_id, p))
        conn.commit()
        print(f"Successfully added '{tag}'!")
    except sqlite3.IntegrityError:
        print(f"Error: Tag '{tag}' already exists.")
    conn.close()

def delete_intent_flow():
    tag = input("Enter the tag name you want to delete: ").strip().lower()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # SQLite doesn't always enable foreign key deletes by default, 
    # so we delete patterns first then the intent.
    cursor.execute("SELECT id FROM intents WHERE tag=?", (tag,))
    result = cursor.fetchone()
    if result:
        cursor.execute("DELETE FROM patterns WHERE intent_id=?", (result[0],))
        cursor.execute("DELETE FROM intents WHERE id=?", (result[0],))
        conn.commit()
        print(f"Deleted intent '{tag}' and all associated patterns.")
    else:
        print("Tag not found.")
    conn.close()

if __name__ == "__main__":
    menu()
