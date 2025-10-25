from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from app.routers.auth import get_current_user
from app.models import User, UserResponse

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at
    )

@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    """A protected route example"""
    return {
        "message": "This is a protected route", 
        "user": {
            "id": str(current_user.id),
            "username": current_user.username,
            "email": current_user.email
        }
    }

@router.get("/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get detailed user profile"""
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at
    }

