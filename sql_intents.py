import sqlite3,re
from deep_translator import GoogleTranslator

DB_NAME = 'chatbot_data.db'

IDIOM_MAP = {
    "so long": "goodbye",
    "piece of cake": "easy",
    "break a leg": "good luck",
    "hit the hay": "go to sleep",
    "under the weather": "sick",
    # Add more as needed
}

import sqlite3
import re
from deep_translator import GoogleTranslator

def translate_enhance_flow():
    print("\n--- Starting Translation Enhancement (with Idiom Support) ---")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get all intents
    cursor.execute("SELECT id, tag FROM intents")
    intents = cursor.fetchall()

    # Added time-out to prevent indefinite hanging
    translator_ja = GoogleTranslator(source='en', target='ja')
    translator_zh = GoogleTranslator(source='en', target='zh-TW')

    added_count = 0

    for intent_id, tag in intents:
        # Fetch existing patterns to check for duplicates
        cursor.execute("SELECT pattern_text FROM patterns WHERE intent_id=?", (intent_id,))
        existing_patterns = [row[0] for row in cursor.fetchall()]

        new_translations = []
        
        for p in existing_patterns:
            # Skip if it's already CJK (Japanese/Chinese characters)
            if not re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', p):
                
                clean_p = p.lower().strip()
                translation_input = IDIOM_MAP.get(clean_p, p)

                try:
                    # Provide feedback so you know the script is alive
                    print(f"  [Translating] '{p}'...", end="\r") 
                    
                    ja_text = translator_ja.translate(translation_input)
                    zh_text = translator_zh.translate(translation_input)
                    
                    if ja_text: new_translations.append(ja_text)
                    if zh_text: new_translations.append(zh_text)
                    
                except Exception as e:
                    print(f"\n  [!] Error translating '{p}': {e}")
                    continue # Move to the next pattern instead of crashing

        # Insert new patterns, avoiding duplicates
        for t_text in set(new_translations):
            if t_text and t_text not in existing_patterns:
                cursor.execute(
                    "INSERT INTO patterns (intent_id, pattern_text) VALUES (?, ?)", 
                    (intent_id, t_text)
                )
                added_count += 1

    conn.commit()
    conn.close()
    print(f"\n\nSuccess! Added {added_count} new patterns (idioms handled).")

def setup_database():
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()

    # 1. Create/Update the intents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT UNIQUE,
            responses TEXT,
            action TEXT,      -- New Column
            parameter TEXT    -- New Column
        )
    ''')

    # 2. Migration: Add columns if they don't exist in an old database
    cursor.execute("PRAGMA table_info(intents)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'action' not in columns:
        cursor.execute('ALTER TABLE intents ADD COLUMN action TEXT')
    if 'parameter' not in columns:
        cursor.execute('ALTER TABLE intents ADD COLUMN parameter TEXT')

    # 3. Create/Update the patterns table (remains the same)
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
    print("Database synced successfully.")

def menu():
    while True:
        print("\n--- CHATBOT KNOWLEDGE MANAGER ---")
        print("1. Create Database")
        print("2. View All Intents")
        print("3. Add New Intent (Tag, Patterns, Responses)")
        print("4. Delete an Intent")
        print("5. Translation Enhancement (Auto-generate CJK patterns)")
        print("6. Exit")
        
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
            translate_enhance_flow()
        elif choice == '6':
            break
        else:
            print("Invalid choice, try again.")

def view_intents():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Update query to include new columns: action and parameter
    cursor.execute("SELECT id, tag, responses, action, parameter FROM intents")
    rows = cursor.fetchall()

    if not rows:
        print("\n[!] The database is empty.")
    else:
        print("\n--- Current Chatbot Intents ---")
        for row in rows:
            intent_id, tag, responses, action, parameter = row
            
            print(f"\nID: {intent_id} | TAG: {tag.upper()}")
            
            # 2. Get patterns for this intent
            cursor.execute("SELECT pattern_text FROM patterns WHERE intent_id=?", (intent_id,))
            patterns = [p[0] for p in cursor.fetchall()]
            
            print(f"  > Patterns:  {', '.join(patterns)}")
            print(f"  > Responses: {responses.replace('|', ' / ')}")
            
            # 3. Display Action/Parameter only if they exist
            if action:
                print(f"  [*] ACTION:    {action}")
            if parameter:
                print(f"  [*] PARAMETER: {parameter}")
                
        print("\n" + "-"*30)
    
    conn.close()

def add_intent_flow():
    tag = input("Enter intent tag (e.g., 'get_weather'): ").strip().lower()
    patterns_raw = input("Enter patterns separated by commas: ")
    responses_raw = input("Enter responses separated by commas: ")
    
    # New prompts for the added columns
    action = input("Enter action name (press Enter if none): ").strip() or None
    parameter = input("Enter parameter (press Enter if none): ").strip() or None

    patterns = [p.strip() for p in patterns_raw.split(',') if p.strip()]
    responses = [r.strip() for r in responses_raw.split(',') if r.strip()]

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # 1. Insert into intents table including action and parameter
        cursor.execute(
            "INSERT INTO intents (tag, responses, action, parameter) VALUES (?, ?, ?, ?)", 
            (tag, "|".join(responses), action, parameter)
        )
        
        intent_id = cursor.lastrowid
        
        # 2. Insert the patterns linked to the new intent_id
        for p in patterns:
            cursor.execute(
                "INSERT INTO patterns (intent_id, pattern_text) VALUES (?, ?)", 
                (intent_id, p)
            )
            
        conn.commit()
        print(f"\nSuccessfully added '{tag}' with action '{action}'!")
        
    except sqlite3.IntegrityError:
        print(f"\nError: Tag '{tag}' already exists in the database.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
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
