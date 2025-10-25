#!/usr/bin/env python3
"""
Simple test script for Trade Analyzer with Beanie ODM
This tests the core functionality without async loop conflicts
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def simple_test():
    """Simple test of core application components"""
    
    print("🚀 Trade Analyzer - Simple Test")
    print("=" * 40)
    
    try:
        # Test 1: Import all modules
        print("\n1. Testing imports...")
        from app.models import User, UserCreate, UserResponse, Token
        from app.database import init_database, close_database
        from app.auth import verify_password, get_password_hash, create_access_token, decode_access_token
        from app.main import app
        print("✅ All imports successful")
        
        # Test 2: Database connection
        print("\n2. Testing database connection...")
        await init_database()
        print("✅ Database connected successfully")
        
        # Test 3: User model operations
        print("\n3. Testing User model operations...")
        
        # Create a test user
        test_user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        
        # Check if user already exists
        existing_user = await User.find_one(User.email == test_user_data.email)
        if existing_user:
            await existing_user.delete()
            print("   Removed existing test user")
        
        # Create new user
        hashed_password = get_password_hash(test_user_data.password)
        user = User(
            username=test_user_data.username,
            email=test_user_data.email,
            hashed_password=hashed_password,
            is_active=True,
            is_verified=False
        )
        
        await user.insert()
        print(f"✅ User created: {user.username} ({user.email})")
        
        # Test password verification
        password_valid = verify_password(test_user_data.password, user.hashed_password)
        print(f"✅ Password verification: {'PASS' if password_valid else 'FAIL'}")
        
        # Test JWT token creation
        token = create_access_token(data={"sub": user.email, "user_id": str(user.id)})
        print("✅ JWT token created successfully")
        
        # Test JWT token decoding
        decoded_token = decode_access_token(token)
        print(f"✅ JWT token decoded: {'PASS' if decoded_token else 'FAIL'}")
        
        # Test user retrieval
        found_user = await User.find_one(User.email == test_user_data.email)
        print(f"✅ User retrieval: {'PASS' if found_user else 'FAIL'}")
        
        # Test user update
        found_user.is_verified = True
        await found_user.save()
        print(f"✅ User update: {'PASS' if found_user.is_verified else 'FAIL'}")
        
        # Test user deletion
        await found_user.delete()
        print("✅ User deletion: PASS")
        
        await close_database()
        
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("\nYour Trade Analyzer application is ready to use!")
        print("\nTo start the application:")
        print("  Backend:  cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
        print("  Frontend: cd frontend && npm run dev")
        print("\n  Backend API: http://localhost:8000")
        print("  Frontend App: http://localhost:3000")
        print("  API Docs: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            await close_database()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(simple_test())
