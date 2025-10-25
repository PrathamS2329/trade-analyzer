#!/usr/bin/env python3
"""
Example usage of Beanie ODM User model
This demonstrates how to work with the User schema
"""

import asyncio
from datetime import datetime
from app.models import User, UserCreate
from app.auth import get_password_hash
from app.database import init_database, close_database

async def example_user_operations():
    """Example of user operations with Beanie ODM"""
    
    # Initialize database
    await init_database()
    
    try:
        # Create a new user
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create User document
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            is_active=True,
            is_verified=False
        )
        
        # Insert user
        await user.insert()
        print(f"✅ User created: {user.username} ({user.email})")
        print(f"   ID: {user.id}")
        print(f"   Created: {user.created_at}")
        
        # Find user by email
        found_user = await User.find_one(User.email == "test@example.com")
        if found_user:
            print(f"✅ User found: {found_user.username}")
        
        # Find user by username
        found_by_username = await User.find_one(User.username == "testuser")
        if found_by_username:
            print(f"✅ User found by username: {found_by_username.email}")
        
        # Update user
        found_user.is_verified = True
        await found_user.save()
        print(f"✅ User updated: verified = {found_user.is_verified}")
        
        # List all users
        all_users = await User.find_all().to_list()
        print(f"✅ Total users: {len(all_users)}")
        
        # Delete user
        await found_user.delete()
        print("✅ User deleted")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Close database connection
        await close_database()

if __name__ == "__main__":
    print("Beanie ODM User Model Example")
    print("=" * 40)
    asyncio.run(example_user_operations())
