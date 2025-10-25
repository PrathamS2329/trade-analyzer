from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

from app.models import User

load_dotenv()

client: AsyncIOMotorClient = None
database = None

async def init_database():
    """Initialize Beanie with MongoDB"""
    global client, database
    
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "trade_analyzer")
    
    # Create motor client
    client = AsyncIOMotorClient(mongodb_url)
    database = client[database_name]
    
    # Initialize Beanie
    await init_beanie(
        database=database,
        document_models=[User]
    )
    
    return database

async def close_database():
    """Close database connection"""
    global client
    if client:
        client.close()

async def get_database():
    """Get database instance"""
    global database
    if database is None:
        await init_database()
    return database

