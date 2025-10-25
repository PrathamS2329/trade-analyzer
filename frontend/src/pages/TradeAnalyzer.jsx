import React, { useState } from 'react'
import PlayerSearch from '../components/PlayerSearch'
import PlayerCard from '../components/PlayerCard'
import './TradeAnalyzer.css'

const TradeAnalyzer = () => {
  const [sideA, setSideA] = useState([])
  const [sideB, setSideB] = useState([])
  const [activeTab, setActiveTab] = useState('A')

  const addPlayerToSide = (side, player) => {
    const newPlayer = { ...player }
    if (side === 'A') {
      setSideA(prev => [...prev, newPlayer])
    } else {
      setSideB(prev => [...prev, newPlayer])
    }
  }

  const removePlayerFromSide = (side, playerId) => {
    if (side === 'A') {
      setSideA(prev => prev.filter(player => player.id !== playerId))
    } else {
      setSideB(prev => prev.filter(player => player.id !== playerId))
    }
  }

  const clearSide = (side) => {
    if (side === 'A') {
      setSideA([])
    } else {
      setSideB([])
    }
  }

  const calculateTotalPoints = (players) => {
    return players.reduce((total, player) => total + player.fantasy_points, 0)
  }

  const getTradeAnalysis = () => {
    const pointsA = calculateTotalPoints(sideA)
    const pointsB = calculateTotalPoints(sideB)
    const difference = Math.abs(pointsA - pointsB)
    
    if (pointsA === pointsB) {
      return { status: 'fair', message: 'Fair trade', color: '#28a745' }
    } else if (difference <= 5) {
      return { status: 'close', message: 'Close trade', color: '#ffc107' }
    } else {
      const winner = pointsA > pointsB ? 'Side A' : 'Side B'
      return { 
        status: 'unfair', 
        message: `${winner} wins by ${difference.toFixed(1)} points`, 
        color: '#dc3545' 
      }
    }
  }

  const analysis = getTradeAnalysis()

  return (
    <div className="trade-analyzer">
      <div className="analyzer-header">
        <h1>Fantasy Basketball Trade Analyzer</h1>
        <p>Search for players and add them to each side to analyze your trade</p>
      </div>

      <div className="analyzer-content">
        <div className="trade-sides">
          <div className="trade-side">
            <div className="side-header">
              <h2>Side A</h2>
              <div className="side-actions">
                <span className="side-count">{sideA.length} player{sideA.length !== 1 ? 's' : ''}</span>
                <button 
                  className="clear-side-btn" 
                  onClick={() => clearSide('A')}
                  disabled={sideA.length === 0}
                >
                  Clear All
                </button>
              </div>
            </div>
            
            <div className="side-total">
              <span className="total-label">Total Points:</span>
              <span className="total-points">{calculateTotalPoints(sideA).toFixed(1)}</span>
            </div>

            <div className="side-players">
              {sideA.length === 0 ? (
                <div className="empty-state">
                  <p>No players added yet</p>
                  <p className="empty-hint">Use the search below to add players</p>
                </div>
              ) : (
                sideA.map(player => (
                  <PlayerCard
                    key={player.id}
                    player={player}
                    onRemove={(playerId) => removePlayerFromSide('A', playerId)}
                    showRemoveButton={true}
                  />
                ))
              )}
            </div>
          </div>

          <div className="trade-vs">
            <div className="vs-divider">
              <span className="vs-text">VS</span>
            </div>
            <div className="trade-analysis">
              <div className="analysis-result" style={{ color: analysis.color }}>
                <div className="analysis-status">{analysis.message}</div>
                <div className="analysis-details">
                  {sideA.length > 0 && sideB.length > 0 && (
                    <span>
                      {Math.abs(calculateTotalPoints(sideA) - calculateTotalPoints(sideB)).toFixed(1)} point difference
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="trade-side">
            <div className="side-header">
              <h2>Side B</h2>
              <div className="side-actions">
                <span className="side-count">{sideB.length} player{sideB.length !== 1 ? 's' : ''}</span>
                <button 
                  className="clear-side-btn" 
                  onClick={() => clearSide('B')}
                  disabled={sideB.length === 0}
                >
                  Clear All
                </button>
              </div>
            </div>
            
            <div className="side-total">
              <span className="total-label">Total Points:</span>
              <span className="total-points">{calculateTotalPoints(sideB).toFixed(1)}</span>
            </div>

            <div className="side-players">
              {sideB.length === 0 ? (
                <div className="empty-state">
                  <p>No players added yet</p>
                  <p className="empty-hint">Use the search below to add players</p>
                </div>
              ) : (
                sideB.map(player => (
                  <PlayerCard
                    key={player.id}
                    player={player}
                    onRemove={(playerId) => removePlayerFromSide('B', playerId)}
                    showRemoveButton={true}
                  />
                ))
              )}
            </div>
          </div>
        </div>

        <div className="player-search-section">
          <div className="search-tabs">
            <button 
              className={`search-tab ${activeTab === 'A' ? 'active' : ''}`}
              onClick={() => setActiveTab('A')}
            >
              Add to Side A
            </button>
            <button 
              className={`search-tab ${activeTab === 'B' ? 'active' : ''}`}
              onClick={() => setActiveTab('B')}
            >
              Add to Side B
            </button>
          </div>
          
          <PlayerSearch 
            onPlayerSelect={(player) => {
              addPlayerToSide(activeTab, player)
            }}
          />
        </div>
      </div>
    </div>
  )
}

export default TradeAnalyzer
