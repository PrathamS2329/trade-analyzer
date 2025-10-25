import api from './authService'

export const playerService = {
  async getAllPlayers(skip = 0, limit = 100) {
    const response = await api.get(`/players/?skip=${skip}&limit=${limit}`)
    return response.data
  },

  async searchPlayers(query, position = null, team = null) {
    const params = new URLSearchParams({ q: query })
    if (position) params.append('position', position)
    if (team) params.append('team', team)
    
    const response = await api.get(`/players/search?${params}`)
    return response.data
  },

  async getPlayer(playerId) {
    const response = await api.get(`/players/${playerId}`)
    return response.data
  },

  async createPlayer(playerData) {
    const response = await api.post('/players/', playerData)
    return response.data
  },

  async getPositions() {
    const response = await api.get('/players/positions/list')
    return response.data
  },

  async getTeams() {
    const response = await api.get('/players/teams/list')
    return response.data
  }
}
