import psycopg2
import schedule
import time
import os
def check_for_changes():
    # Connect to the database and query for changes
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute("SELECT * FROM your_table WHERE modified_at > %s", (last_check_time,))
    changes = cur.fetchall()
    cur.close()
    conn.close()

    # Process the changes
    for change in changes:
        print(change)

# Schedule the check_for_changes function to run every minute
schedule.every(1).minutes.do(check_for_changes)

while True:
    schedule.run_pending()
    time.sleep(1)