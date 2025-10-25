import React from 'react'
import './PlayerCard.css'

const PlayerCard = ({ player, onRemove, showRemoveButton = true }) => {
  const handleRemove = () => {
    if (onRemove) {
      onRemove(player.id)
    }
  }

  return (
    <div className="player-card">
      <div className="player-info">
        <div className="player-name">{player.name}</div>
        <div className="player-details">
          <span className="player-position">{player.position}</span>
          <span className="player-team">{player.team}</span>
        </div>
        <div className="player-stats">
          <div className="fantasy-points">
            <span className="points-label">Fantasy Points:</span>
            <span className="points-value">{player.fantasy_points.toFixed(1)}</span>
          </div>
        </div>
      </div>
      {showRemoveButton && (
        <button className="remove-button" onClick={handleRemove}>
          ×
        </button>
      )}
    </div>
  )
}

export default PlayerCard
