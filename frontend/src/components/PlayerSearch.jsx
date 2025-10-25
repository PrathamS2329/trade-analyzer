import React, { useState, useEffect } from 'react'
import { playerService } from '../services/playerService'
import './PlayerSearch.css'

const PlayerSearch = ({ onPlayerSelect }) => {
  const [query, setQuery] = useState('')
  const [position, setPosition] = useState('')
  const [team, setTeam] = useState('')
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(false)
  const [positions, setPositions] = useState([])
  const [teams, setTeams] = useState([])

  useEffect(() => {
    // Load positions and teams when component mounts
    const loadFilters = async () => {
      try {
        const [positionsRes, teamsRes] = await Promise.all([
          playerService.getPositions(),
          playerService.getTeams()
        ])
        setPositions(positionsRes.positions)
        setTeams(teamsRes.teams)
      } catch (error) {
        console.error('Failed to load filters:', error)
      }
    }
    loadFilters()
  }, [])

  useEffect(() => {
    // Search players when query changes
    if (query.length >= 2) {
      searchPlayers()
    } else {
      setPlayers([])
    }
  }, [query, position, team])

  const searchPlayers = async () => {
    if (query.length < 2) return

    setLoading(true)
    try {
      const results = await playerService.searchPlayers(query, position, team)
      setPlayers(results)
    } catch (error) {
      console.error('Search failed:', error)
      setPlayers([])
    } finally {
      setLoading(false)
    }
  }

  const handlePlayerClick = (player) => {
    onPlayerSelect(player)
    setQuery('') // Clear search after selection
    setPlayers([])
  }

  const clearFilters = () => {
    setQuery('')
    setPosition('')
    setTeam('')
    setPlayers([])
  }

  return (
    <div className="player-search">
      <div className="search-header">
        <h3>Search Players</h3>
        <button className="clear-filters" onClick={clearFilters}>
          Clear Filters
        </button>
      </div>
      
      <div className="search-form">
        <div className="search-input-group">
          <input
            type="text"
            placeholder="Search by player name..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="search-input"
          />
        </div>
        
        <div className="filter-group">
          <select
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            className="filter-select"
          >
            <option value="">All Positions</option>
            {positions.map(pos => (
              <option key={pos} value={pos}>{pos}</option>
            ))}
          </select>
          
          <select
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            className="filter-select"
          >
            <option value="">All Teams</option>
            {teams.map(teamName => (
              <option key={teamName} value={teamName}>{teamName}</option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <span>Searching...</span>
        </div>
      )}

      {players.length > 0 && (
        <div className="search-results">
          <div className="results-header">
            <span>Found {players.length} player{players.length !== 1 ? 's' : ''}</span>
          </div>
          <div className="results-list">
            {players.map(player => (
              <div
                key={player.id}
                className="search-result-item"
                onClick={() => handlePlayerClick(player)}
              >
                <div className="result-player-name">{player.name}</div>
                <div className="result-player-details">
                  <span className="result-position">{player.position}</span>
                  <span className="result-team">{player.team}</span>
                  <span className="result-points">{player.fantasy_points.toFixed(1)} pts</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {query.length >= 2 && players.length === 0 && !loading && (
        <div className="no-results">
          <p>No players found matching your search.</p>
          <p className="no-results-hint">Try adjusting your filters or search terms.</p>
        </div>
      )}
    </div>
  )
}

export default PlayerSearch
