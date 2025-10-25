# Trade Analyzer Backend

FastAPI backend with MongoDB and JWT authentication using Beanie ODM.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

3. Update `.env` with your MongoDB connection string and secret key.

4. Make sure MongoDB is running on your machine or update the `MONGODB_URL` in `.env`.

5. Run the application:
```bash
uvicorn app.main:app --reload
```

## Features

- **Beanie ODM**: Modern MongoDB ODM with Pydantic integration
- **User Schema**: Structured user model with validation
- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt password security
- **Async/Await**: Full async support

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get JWT token

### Users
- `GET /api/users/me` - Get current user info (requires authentication)
- `GET /api/users/protected` - Protected route example (requires authentication)
- `GET /api/users/profile` - Get detailed user profile (requires authentication)

## User Schema

The User model includes:
- `username` - Unique username (indexed)
- `email` - Unique email address (indexed)
- `hashed_password` - bcrypt hashed password
- `is_active` - Account status
- `is_verified` - Email verification status
- `created_at` - Account creation timestamp
- `updated_at` - Last update timestamp

## Environment Variables

- `MONGODB_URL`: MongoDB connection string
- `DATABASE_NAME`: Database name
- `SECRET_KEY`: Secret key for JWT tokens

## Database Features

- **Automatic Indexing**: Username and email are automatically indexed
- **Validation**: Pydantic validation for all fields
- **Type Safety**: Full type hints and validation
- **Async Operations**: Non-blocking database operations

