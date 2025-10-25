#!/bin/bash

echo "🚀 Setting up Trade Analyzer with Beanie ODM..."

# Backend setup
echo ""
echo "📦 Setting up backend..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies (including Beanie ODM)..."
pip install -r requirements.txt

echo "Testing Beanie setup..."
python3 test_setup.py

echo "Backend setup complete! ✅"
cd ..

# Frontend setup
echo ""
echo "📦 Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

echo "Frontend setup complete! ✅"
cd ..

echo ""
echo "✨ Setup complete!"
echo ""
echo "New Features with Beanie ODM:"
echo "  - Structured User schema with validation"
echo "  - Automatic database indexing"
echo "  - Type-safe database operations"
echo "  - Pydantic integration"
echo ""
echo "To run the application:"
echo "  Terminal 1 (Backend):  cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "  Terminal 2 (Frontend): cd frontend && npm run dev"
echo ""
echo "Make sure MongoDB is running!"
echo "Backend API: http://localhost:8000"
echo "Frontend App: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Example User Operations:"
echo "  cd backend && source venv/bin/activate && python3 example_user_operations.py"

