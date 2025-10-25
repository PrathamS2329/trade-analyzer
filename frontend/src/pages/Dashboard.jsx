import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { authService } from '../services/authService'
import './Dashboard.css'

const Dashboard = () => {
  const { logout, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const [userInfo, setUserInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login')
      return
    }

    const fetchUserInfo = async () => {
      try {
        const response = await authService.getCurrentUser()
        setUserInfo(response.user)
      } catch (error) {
        console.error('Failed to fetch user info:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchUserInfo()
  }, [isAuthenticated, navigate])

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  if (loading) {
    return <div className="loading">Loading...</div>
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-card">
        <div className="dashboard-header">
          <div className="welcome-section">
            <h1>Welcome to Trade Analyzer</h1>
            {userInfo && (
              <h2 className="username-display">
                Hello, <span className="username-highlight">{userInfo.username}</span>!
              </h2>
            )}
          </div>
          <button onClick={handleLogout} className="logout-button">
            Logout
          </button>
        </div>
        <div className="user-info">
          <h3>Account Details</h3>
          {userInfo && (
            <div className="info-grid">
              <div className="info-item">
                <strong>Username:</strong> {userInfo.username}
              </div>
              <div className="info-item">
                <strong>Email:</strong> {userInfo.email}
              </div>
              <div className="info-item">
                <strong>Status:</strong> 
                <span className={`status-badge ${userInfo.is_active ? 'active' : 'inactive'}`}>
                  {userInfo.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="info-item">
                <strong>Verified:</strong> 
                <span className={`status-badge ${userInfo.is_verified ? 'verified' : 'unverified'}`}>
                  {userInfo.is_verified ? 'Verified' : 'Unverified'}
                </span>
              </div>
            </div>
          )}
        </div>
        <div className="dashboard-content">
          <div className="feature-cards">
            <div className="feature-card" onClick={() => navigate('/trade-analyzer')}>
              <div className="feature-icon">🏀</div>
              <h3>Trade Analyzer</h3>
              <p>Analyze fantasy basketball trades with player comparisons and point calculations</p>
              <button className="feature-button">Start Analyzing</button>
            </div>
            <div className="feature-card coming-soon">
              <div className="feature-icon">📊</div>
              <h3>Player Stats</h3>
              <p>View detailed player statistics and performance metrics</p>
              <button className="feature-button" disabled>Coming Soon</button>
            </div>
            <div className="feature-card coming-soon">
              <div className="feature-icon">📈</div>
              <h3>Trends</h3>
              <p>Track player trends and market values over time</p>
              <button className="feature-button" disabled>Coming Soon</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard

