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
          <h1>Welcome to Trade Analyzer</h1>
          <button onClick={handleLogout} className="logout-button">
            Logout
          </button>
        </div>
        <div className="user-info">
          <h2>User Information</h2>
          {userInfo && (
            <div className="info-item">
              <strong>Email:</strong> {userInfo.email}
            </div>
          )}
        </div>
        <div className="dashboard-content">
          <p>This is your protected dashboard. You can start building your trade analyzer features here.</p>
        </div>
      </div>
    </div>
  )
}

export default Dashboard

