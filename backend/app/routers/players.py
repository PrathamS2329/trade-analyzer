from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPAuthorizationCredentials
from typing import List, Optional
from app.routers.auth import get_current_user
from app.models import User, Player, PlayerCreate, PlayerResponse, PlayerSearch

router = APIRouter()

@router.get("/", response_model=List[PlayerResponse])
async def get_all_players(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user)
):
    """Get all players with pagination"""
    players = await Player.find_all().skip(skip).limit(limit).to_list()
    return [PlayerResponse(
        id=str(player.id),
        name=player.name,
        position=player.position,
        team=player.team,
        stats=player.stats,
        fantasy_points=player.fantasy_points,
        is_active=player.is_active,
        created_at=player.created_at
    ) for player in players]

@router.get("/search", response_model=List[PlayerResponse])
async def search_players(
    q: str = Query(..., min_length=1),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Search players by name, position, or team"""
    query_filters = []
    
    # Search by name (case-insensitive)
    if q:
        query_filters.append({"name": {"$regex": q, "$options": "i"}})
    
    # Filter by position
    if position:
        query_filters.append({"position": position})
    
    # Filter by team
    if team:
        query_filters.append({"team": {"$regex": team, "$options": "i"}})
    
    # Only show active players
    query_filters.append({"is_active": True})
    
    if query_filters:
        players = await Player.find({"$and": query_filters}).limit(50).to_list()
    else:
        players = await Player.find({"is_active": True}).limit(50).to_list()
    
    return [PlayerResponse(
        id=str(player.id),
        name=player.name,
        position=player.position,
        team=player.team,
        stats=player.stats,
        fantasy_points=player.fantasy_points,
        is_active=player.is_active,
        created_at=player.created_at
    ) for player in players]

@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(
    player_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific player by ID"""
    try:
        from bson import ObjectId
        player = await Player.get(ObjectId(player_id))
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Player not found"
            )
        
        return PlayerResponse(
            id=str(player.id),
            name=player.name,
            position=player.position,
            team=player.team,
            stats=player.stats,
            fantasy_points=player.fantasy_points,
            is_active=player.is_active,
            created_at=player.created_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid player ID"
        )

@router.post("/", response_model=PlayerResponse)
async def create_player(
    player_data: PlayerCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new player (admin function)"""
    # Check if player already exists
    existing_player = await Player.find_one({"name": player_data.name, "team": player_data.team})
    if existing_player:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Player already exists"
        )
    
    player = Player(
        name=player_data.name,
        position=player_data.position,
        team=player_data.team,
        stats=player_data.stats,
        fantasy_points=player_data.fantasy_points,
        is_active=True
    )
    
    await player.insert()
    
    return PlayerResponse(
        id=str(player.id),
        name=player.name,
        position=player.position,
        team=player.team,
        stats=player.stats,
        fantasy_points=player.fantasy_points,
        is_active=player.is_active,
        created_at=player.created_at
    )

@router.get("/positions/list")
async def get_positions(current_user: User = Depends(get_current_user)):
    """Get list of all unique positions"""
    positions = await Player.find({"is_active": True}).distinct("position")
    return {"positions": sorted(positions)}

@router.get("/teams/list")
async def get_teams(current_user: User = Depends(get_current_user)):
    """Get list of all unique teams"""
    teams = await Player.find({"is_active": True}).distinct("team")
    return {"teams": sorted(teams)}
