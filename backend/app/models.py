from typing import Optional, List
from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from bson import ObjectId

class User(Document):
    """User document model using Beanie ODM"""
    username: Indexed(str, unique=True)
    email: Indexed(EmailStr, unique=True)
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "users"
        indexes = [
            "username",
            "email",
        ]

class Player(Document):
    """Player document model for fantasy basketball"""
    name: Indexed(str)
    position: str
    team: str
    stats: dict = Field(default_factory=dict)
    fantasy_points: float = 0.0
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "players"
        indexes = [
            "name",
            "position",
            "team",
        ]

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=72)  # bcrypt limit

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class PlayerCreate(BaseModel):
    name: str
    position: str
    team: str
    stats: dict = Field(default_factory=dict)
    fantasy_points: float = 0.0

class PlayerResponse(BaseModel):
    id: str
    name: str
    position: str
    team: str
    stats: dict
    fantasy_points: float
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class PlayerSearch(BaseModel):
    query: str = Field(..., min_length=1)
    position: Optional[str] = None
    team: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None

