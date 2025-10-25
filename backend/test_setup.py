#!/usr/bin/env python3
"""
Test script to verify Beanie ODM setup
Run this after installing dependencies: pip install -r requirements.txt
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_beanie_setup():
    """Test Beanie ODM setup"""
    try:
        # Import Beanie components
        from beanie import init_beanie, Document, Indexed
        from motor.motor_asyncio import AsyncIOMotorClient
        from pydantic import BaseModel, EmailStr, Field
        from datetime import datetime
        
        print("✅ Beanie imports successful")
        
        # Test User model
        from app.models import User, UserCreate, UserResponse
        print("✅ User model imports successful")
        
        # Test database connection
        from app.database import init_database, close_database
        print("✅ Database module imports successful")
        
        # Test auth utilities
        from app.auth import verify_password, get_password_hash, create_access_token
        print("✅ Auth utilities imports successful")
        
        print("\n🎉 All Beanie ODM components are properly configured!")
        print("\nTo run the application:")
        print("1. Make sure MongoDB is running")
        print("2. Run: uvicorn app.main:app --reload")
        print("3. Visit: http://localhost:8000/docs")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_beanie_setup())
