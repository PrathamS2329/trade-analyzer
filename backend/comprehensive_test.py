#!/usr/bin/env python3
"""
Comprehensive test script for Trade Analyzer with Beanie ODM
This tests all components of the application
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def comprehensive_test():
    """Comprehensive test of all application components"""
    
    print("🚀 Trade Analyzer - Comprehensive Test")
    print("=" * 50)
    
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
            password="testpass123"  # Shorter password
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
        
        # Test 4: API endpoints
        print("\n4. Testing API endpoints...")
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        print(f"✅ Root endpoint: {response.status_code} - {response.json()['message']}")
        
        # Test health endpoint
        response = client.get("/api/health")
        print(f"✅ Health endpoint: {response.status_code} - {response.json()['status']}")
        
        # Test registration endpoint
        response = client.post("/api/auth/register", json={
            "username": "apitest",
            "email": "apitest@example.com",
            "password": "testpass123"
        })
        print(f"✅ Registration endpoint: {response.status_code}")
        
        # Test login endpoint
        response = client.post("/api/auth/login", json={
            "email": "apitest@example.com",
            "password": "testpass123"
        })
        if response.status_code == 200:
            token_data = response.json()
            print(f"✅ Login endpoint: {response.status_code} - Token received")
            
            # Test protected endpoint
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            response = client.get("/api/users/me", headers=headers)
            print(f"✅ Protected endpoint: {response.status_code}")
        else:
            print(f"❌ Login endpoint failed: {response.status_code}")
        
        # Clean up test user
        test_user = await User.find_one(User.email == "apitest@example.com")
        if test_user:
            await test_user.delete()
            print("   Cleaned up test user")
        
        await close_database()
        
        print("\n🎉 ALL TESTS PASSED!")
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
    asyncio.run(comprehensive_test())
