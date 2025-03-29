import os
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "career-predictor"
COLLECTION_NAME = "career_insights"

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Load data from CSV
csv_file = "../career_data.csv"  # Adjust the path if needed
df = pd.read_csv(csv_file)

# Convert DataFrame to dictionary and insert into MongoDB
data = df.to_dict(orient="records")
collection.insert_many(data)

print("Data inserted successfully into MongoDB Atlas!")
