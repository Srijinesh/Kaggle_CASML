import os
import json

def create_single_query():
    queries_file = "Data/queries.json"
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Take exactly one query
    sample = [data[0]]
    
    with open("Data/single_query.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=4)

if __name__ == "__main__":
    create_single_query()
