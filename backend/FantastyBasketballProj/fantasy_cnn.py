"""
Fantasy Basketball CNN Trade Analyzer
Complete implementation with feature engineering and model architecture
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


# ============================================================================
# PART 1: FEATURE ENGINEERING PIPELINE
# ============================================================================

class FantasyFeatureEngine:
    """
    Comprehensive feature engineering for fantasy basketball predictions
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def calculate_injury_risk_score(self, player_data: Dict) -> float:
        """
        Calculate injury risk score (0-1 scale)
        """
        games_missed_last = player_data.get('games_missed_last_season', 0)
        games_missed_current = player_data.get('games_missed_current', 0)
        games_played_current = player_data.get('games_played', 1)
        age = player_data.get('age', 25)
        current_status = player_data.get('injury_status', 'healthy')  # healthy, questionable, out

        # Age factor
        if age < 27:
            age_factor = 0
        elif age <= 30:
            age_factor = 0.1
        elif age <= 33:
            age_factor = 0.3
        else:
            age_factor = 0.5

        # Current status factor
        status_map = {'healthy': 0, 'day-to-day': 0.3, 'questionable': 0.5, 'out': 1.0}
        status_factor = status_map.get(current_status.lower(), 0)

        # Calculate injury risk
        injury_risk = (
                (games_missed_last / 82) * 0.3 +
                (games_missed_current / max(games_played_current, 1)) * 0.4 +
                age_factor * 0.2 +
                status_factor * 0.1
        )

        return min(injury_risk, 1.0)

    def calculate_schedule_difficulty(self, player_data: Dict) -> float:
        """
        Calculate schedule difficulty score (0-1 scale)
        Lower = easier schedule, Higher = harder schedule
        """
        opponent_def_ratings = player_data.get('next_5_opponent_def_ratings', [0.5] * 5)
        opponent_pace = player_data.get('next_5_opponent_pace', [100] * 5)
        back_to_backs = player_data.get('back_to_backs_remaining', 0)
        total_games = player_data.get('games_remaining', 82)

        # Normalize opponent defensive ratings (assuming 105-115 range)
        avg_def_rating = np.mean(opponent_def_ratings)
        def_rating_normalized = (avg_def_rating - 105) / 10  # 0-1 scale

        # Normalize pace (assuming 95-105 range)
        avg_pace = np.mean(opponent_pace)
        pace_normalized = 1 - ((avg_pace - 95) / 10)  # Inverted: higher pace = easier

        # Back-to-back factor
        b2b_factor = (back_to_backs / max(total_games, 1)) if total_games > 0 else 0

        schedule_difficulty = (
                def_rating_normalized * 0.6 +
                pace_normalized * 0.2 +
                b2b_factor * 0.2
        )

        return np.clip(schedule_difficulty, 0, 1)

    def calculate_teammate_impact_score(self, player_data: Dict) -> float:
        """
        The "Jokic Effect" - how much does having a star teammate help?
        """
        star_teammate_assist_rate = player_data.get('star_teammate_assist_rate', 0)
        star_teammate_usage = player_data.get('star_teammate_usage_rate', 0)
        star_on_off_split = player_data.get('fp_with_star_vs_without', 0)  # FP difference
        double_team_gravity = player_data.get('star_teammate_double_team_rate', 0)

        # Positive impact: better looks from star playmaker
        positive_impact = (
                star_teammate_assist_rate * 0.3 +
                double_team_gravity * 0.3 +
                (star_on_off_split / 50) * 0.4  # Normalize by typical FP
        )

        # Negative impact: reduced usage
        negative_impact = star_teammate_usage * 0.5

        # Net impact (-1 to 1 scale)
        net_impact = positive_impact - negative_impact

        return np.clip(net_impact, -1, 1)

    def calculate_contract_motivation_score(self, player_data: Dict) -> float:
        """
        Contract year and motivation factors
        """
        is_contract_year = player_data.get('contract_year', False)
        years_until_fa = player_data.get('years_until_free_agency', 3)
        is_prove_it_deal = player_data.get('prove_it_deal', False)
        contract_value_remaining = player_data.get('contract_value_remaining_millions', 50)

        motivation_score = 0

        # Contract year boost
        if is_contract_year:
            motivation_score += 0.4

        # Urgency factor
        motivation_score += (1 / max(years_until_fa, 1)) * 0.3

        # Prove-it deal boost
        if is_prove_it_deal:
            motivation_score += 0.2

        # Low contract value = needs to prove worth
        if contract_value_remaining < 20:
            motivation_score += 0.1

        return min(motivation_score, 1.0)

    def calculate_usage_explosion_probability(self, player_data: Dict) -> float:
        """
        Probability of usage rate explosion (star leaves, injury, etc.)
        """
        star_teammate_left = player_data.get('star_teammate_recently_left', False)
        days_since_roster_change = player_data.get('days_since_roster_change', 999)
        shot_attempts_trend = player_data.get('shot_attempts_last_5_vs_season', 1.0)  # ratio
        usage_rate_trend = player_data.get('usage_rate_last_5_vs_season', 1.0)
        new_starting_role = player_data.get('new_starting_role', False)

        explosion_prob = 0

        # Star left = usage spike
        if star_teammate_left and days_since_roster_change < 30:
            explosion_prob += 0.4

        # Shot attempts increasing
        if shot_attempts_trend > 1.1:
            explosion_prob += 0.2

        # Usage rate increasing
        if usage_rate_trend > 1.1:
            explosion_prob += 0.2

        # New starting role
        if new_starting_role:
            explosion_prob += 0.2

        return min(explosion_prob, 1.0)

    def calculate_matchup_advantage_score(self, player_data: Dict) -> float:
        """
        Historical performance vs upcoming opponents
        """
        historical_fp_vs_opponents = player_data.get('historical_fp_vs_next_5_opponents', [])
        season_avg_fp = player_data.get('season_avg_fp', 30)

        if not historical_fp_vs_opponents:
            return 0.0

        # Calculate advantage: (avg vs opponents - season avg) / season avg
        avg_vs_opponents = np.mean(historical_fp_vs_opponents)
        advantage = (avg_vs_opponents - season_avg_fp) / max(season_avg_fp, 1)

        # Normalize to -1 to 1 scale
        return np.clip(advantage, -1, 1)

    def calculate_breakout_probability(self, player_data: Dict) -> float:
        """
        Young player breakout detection
        """
        age = player_data.get('age', 30)
        minutes_trend = player_data.get('minutes_last_10_vs_previous_10', 1.0)  # ratio
        usage_trend = player_data.get('usage_rate_last_10_vs_previous_10', 1.0)
        fp_trend = player_data.get('fp_last_10_vs_previous_10', 1.0)
        is_tanking_team = player_data.get('team_is_tanking', False)

        breakout_prob = 0

        # Young player bonus
        if age < 25:
            breakout_prob += 0.3

        # Increasing minutes
        if minutes_trend > 1.15:
            breakout_prob += 0.25

        # Increasing usage
        if usage_trend > 1.1:
            breakout_prob += 0.2

        # Performance trending up
        if fp_trend > 1.1:
            breakout_prob += 0.15

        # Tanking team gives opportunities
        if is_tanking_team:
            breakout_prob += 0.1

        return min(breakout_prob, 1.0)

    def engineer_all_features(self, player_data: Dict) -> Dict:
        """
        Main method: Engineer ALL features for a player
        """
        features = {}

        # Basic stats (from raw data)
        features['age'] = player_data.get('age', 25)
        features['games_played'] = player_data.get('games_played', 0)
        features['games_remaining'] = player_data.get('games_remaining', 82)
        features['season_avg_fp'] = player_data.get('season_avg_fp', 0)
        features['minutes_per_game'] = player_data.get('mpg', 0)
        features['usage_rate'] = player_data.get('usage_rate', 0)
        features['true_shooting_pct'] = player_data.get('ts_pct', 0.5)
        features['assist_to_turnover'] = player_data.get('ast_to_ratio', 1.0)
        features['per'] = player_data.get('per', 15)

        # Performance trends
        features['last_5_avg_fp'] = player_data.get('last_5_avg_fp', 0)
        features['last_10_avg_fp'] = player_data.get('last_10_avg_fp', 0)
        features['last_15_avg_fp'] = player_data.get('last_15_avg_fp', 0)
        features['fp_std_dev_15'] = player_data.get('fp_std_15', 5)
        features['fp_std_dev_30'] = player_data.get('fp_std_30', 5)

        # Calculated advanced features
        features['injury_risk_score'] = self.calculate_injury_risk_score(player_data)
        features['schedule_difficulty'] = self.calculate_schedule_difficulty(player_data)
        features['teammate_impact_score'] = self.calculate_teammate_impact_score(player_data)
        features['contract_motivation_score'] = self.calculate_contract_motivation_score(player_data)
        features['usage_explosion_prob'] = self.calculate_usage_explosion_probability(player_data)
        features['matchup_advantage_score'] = self.calculate_matchup_advantage_score(player_data)
        features['breakout_probability'] = self.calculate_breakout_probability(player_data)

        # Team context
        features['team_pace'] = player_data.get('team_pace', 100)
        features['team_win_pct'] = player_data.get('team_win_pct', 0.5)
        features['is_starter'] = float(player_data.get('is_starter', True))

        # Position (one-hot encoded)
        position = player_data.get('position', 'SG')
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            features[f'pos_{pos}'] = float(pos in position)

        # Volatility metrics
        features['coefficient_variation'] = features['fp_std_dev_15'] / max(features['season_avg_fp'], 1)
        features['floor_game_pct'] = player_data.get('pct_games_below_20fp', 0)
        features['ceiling_game_pct'] = player_data.get('pct_games_above_50fp', 0)

        return features

    def calculate_high_low_projections(self, features: Dict) -> Tuple[float, float, float]:
        """
        Calculate High End, Low End, and Expected Average using our formula
        """
        base_avg = features['season_avg_fp']
        std_dev = features['fp_std_dev_15']
        injury_risk = features['injury_risk_score']
        schedule_diff = features['schedule_difficulty']

        # Trend bonus/penalty
        recent_avg = features['last_5_avg_fp']
        trend_diff = recent_avg - base_avg

        # HIGH END CALCULATION
        high_end = base_avg + (std_dev * 1.5 * 0.15)  # Volatility boost
        high_end += trend_diff * 0.08  # Trend bonus

        # Schedule boost (only if easy schedule)
        if schedule_diff < 0.4:
            high_end += base_avg * 0.05

        # Contract motivation boost
        high_end += base_avg * features['contract_motivation_score'] * 0.03

        # Breakout boost
        high_end += base_avg * features['breakout_probability'] * 0.08

        # Injury risk (minor ceiling impact)
        high_end -= base_avg * injury_risk * 0.05

        # LOW END CALCULATION
        low_end = base_avg - (std_dev * 1.5 * 0.15)  # Volatility penalty
        low_end += trend_diff * 0.08  # Trend penalty (can be negative)

        # Schedule penalty (only if hard schedule)
        if schedule_diff > 0.6:
            low_end -= base_avg * 0.05

        # Injury risk (major floor impact)
        low_end -= base_avg * injury_risk * 0.12

        # Usage explosion probability (raises floor)
        low_end += base_avg * features['usage_explosion_prob'] * 0.05

        # EXPECTED AVERAGE (weighted between current and trends)
        expected_avg = base_avg * 0.7 + recent_avg * 0.3
        expected_avg += base_avg * features['matchup_advantage_score'] * 0.03

        return max(high_end, 0), max(low_end, 0), max(expected_avg, 0)


# ============================================================================
# PART 2: CNN MODEL ARCHITECTURE
# ============================================================================

class FantasyBasketballCNN:
    """
    Hybrid CNN-LSTM model with multi-head attention for fantasy predictions
    """

    def __init__(self, n_features: int = 50, sequence_length: int = 15):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.model = None
        self.feature_engine = FantasyFeatureEngine()

    def build_model(self):
        """
        Build the complete model architecture
        """
        # Input: (batch_size, sequence_length, n_features)
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))

        # ===== BRANCH 1: Conv1D for pattern recognition =====
        conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Dropout(0.2)(conv1)

        conv2 = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(0.3)(conv2)

        conv3 = layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(conv2)
        conv3 = layers.BatchNormalization()(conv3)

        # ===== BRANCH 2: LSTM for temporal dependencies =====
        lstm1 = layers.LSTM(128, return_sequences=True)(input_layer)
        lstm1 = layers.Dropout(0.3)(lstm1)

        lstm2 = layers.LSTM(64, return_sequences=False)(lstm1)
        lstm2 = layers.Dropout(0.3)(lstm2)

        # ===== Attention Mechanism =====
        # Multi-head attention on Conv output
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(conv3, conv3)
        attention = layers.GlobalAveragePooling1D()(attention)

        # ===== Merge branches =====
        conv_flat = layers.Flatten()(conv3)
        merged = layers.Concatenate()([conv_flat, lstm2, attention])

        # ===== Dense layers =====
        dense1 = layers.Dense(256, activation='relu')(merged)
        dense1 = layers.Dropout(0.4)(dense1)

        dense2 = layers.Dense(128, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)

        dense3 = layers.Dense(64, activation='relu')(dense2)

        # ===== Output heads =====
        # Multiple outputs for different predictions
        high_end_output = layers.Dense(1, activation='relu', name='high_end')(dense3)
        low_end_output = layers.Dense(1, activation='relu', name='low_end')(dense3)
        expected_avg_output = layers.Dense(1, activation='relu', name='expected_avg')(dense3)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(dense3)

        # Create model
        self.model = models.Model(
            inputs=input_layer,
            outputs=[high_end_output, low_end_output, expected_avg_output, confidence_output]
        )

        # Compile with custom loss weights
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'high_end': 'mse',
                'low_end': 'mse',
                'expected_avg': 'mse',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={
                'high_end': 1.0,
                'low_end': 1.0,
                'expected_avg': 1.5,  # Weight expected avg more
                'confidence': 0.5
            },
            metrics=['mae']
        )

        return self.model

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Prepare data for training
        df: DataFrame with columns for each game and player stats
        """
        # This will be customized based on actual data format
        # For now, placeholder structure
        X = []
        y_high = []
        y_low = []
        y_expected = []
        y_confidence = []

        # Group by player and create sequences
        for player_id in df['player_id'].unique():
            player_df = df[df['player_id'] == player_id].sort_values('game_date')

            # Create rolling windows
            for i in range(len(player_df) - self.sequence_length):
                sequence = player_df.iloc[i:i + self.sequence_length]

                # Extract features for sequence
                feature_vectors = []
                for _, game in sequence.iterrows():
                    game_dict = game.to_dict()
                    features = self.feature_engine.engineer_all_features(game_dict)
                    feature_vectors.append(list(features.values()))

                X.append(feature_vectors)

                # Target: next game performance
                next_game = player_df.iloc[i + self.sequence_length]
                next_game_dict = next_game.to_dict()
                features = self.feature_engine.engineer_all_features(next_game_dict)
                high, low, expected = self.feature_engine.calculate_high_low_projections(features)

                y_high.append(high)
                y_low.append(low)
                y_expected.append(expected)
                y_confidence.append(1.0)  # Placeholder

        X = np.array(X)
        y = {
            'high_end': np.array(y_high),
            'low_end': np.array(y_low),
            'expected_avg': np.array(y_expected),
            'confidence': np.array(y_confidence)
        }

        return X, y

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model
        """
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)


# ============================================================================
# PART 3: TRADE ANALYZER
# ============================================================================

class TradeAnalyzer:
    """
    Analyze trades and provide recommendations
    """

    def __init__(self, model: FantasyBasketballCNN):
        self.model = model
        self.feature_engine = FantasyFeatureEngine()

    def analyze_trade(self,
                      players_giving: List[Dict],
                      players_receiving: List[Dict],
                      league_context: Dict) -> Dict:
        """
        Analyze a multi-player trade

        Args:
            players_giving: List of player data dicts you're trading away
            players_receiving: List of player data dicts you're getting
            league_context: League settings, roster needs, etc.

        Returns:
            Comprehensive trade analysis
        """
        analysis = {
            'trade_value_difference': 0,
            'risk_assessment': {},
            'roster_fit': {},
            'recommendation': '',
            'confidence': 0
        }

        # Calculate total value for each side
        giving_value = self._calculate_total_value(players_giving)
        receiving_value = self._calculate_total_value(players_receiving)

        analysis['giving_total_value'] = giving_value
        analysis['receiving_total_value'] = receiving_value
        analysis['trade_value_difference'] = receiving_value - giving_value

        # Risk assessment
        analysis['risk_assessment'] = {
            'giving_injury_risk': np.mean([p.get('injury_risk_score', 0) for p in players_giving]),
            'receiving_injury_risk': np.mean([p.get('injury_risk_score', 0) for p in players_receiving]),
            'giving_volatility': np.mean([p.get('fp_std_dev_15', 0) for p in players_giving]),
            'receiving_volatility': np.mean([p.get('fp_std_dev_15', 0) for p in players_receiving])
        }

        # Generate recommendation
        if analysis['trade_value_difference'] > 5:
            analysis['recommendation'] = 'ACCEPT - Good value trade'
            analysis['confidence'] = min(analysis['trade_value_difference'] / 10, 1.0)
        elif analysis['trade_value_difference'] < -5:
            analysis['recommendation'] = 'REJECT - Losing value'
            analysis['confidence'] = min(abs(analysis['trade_value_difference']) / 10, 1.0)
        else:
            analysis['recommendation'] = 'NEUTRAL - Even trade, consider roster fit'
            analysis['confidence'] = 0.5

        return analysis

    def _calculate_total_value(self, players: List[Dict]) -> float:
        """
        Calculate total fantasy value for a group of players
        """
        total_value = 0

        for player in players:
            features = self.feature_engine.engineer_all_features(player)
            high, low, expected = self.feature_engine.calculate_high_low_projections(features)

            # Value = Expected avg weighted by confidence
            confidence = 1 - features['injury_risk_score']
            value = expected * confidence * player.get('games_remaining', 82) / 82

            total_value += value

        return total_value

    def find_trade_targets(self,
                           my_roster: List[Dict],
                           all_players: List[Dict],
                           need_position: str = None) -> List[Dict]:
        """
        Find players to target in trades based on your needs
        """
        targets = []

        for player in all_players:
            # Skip players on my roster
            if player['player_id'] in [p['player_id'] for p in my_roster]:
                continue

            # Filter by position if specified
            if need_position and need_position not in player.get('position', ''):
                continue

            features = self.feature_engine.engineer_all_features(player)
            high, low, expected = self.feature_engine.calculate_high_low_projections(features)

            targets.append({
                'player_name': player['player_name'],
                'expected_fp': expected,
                'upside': high - expected,
                'injury_risk': features['injury_risk_score'],
                'breakout_probability': features['breakout_probability']
            })

        # Sort by expected FP
        targets.sort(key=lambda x: x['expected_fp'], reverse=True)

        return targets[:20]  # Top 20 targets


# ============================================================================
# PART 4: USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use the system
    """
    # Initialize
    feature_engine = FantasyFeatureEngine()

    # Example player data
    player_data = {
        'player_name': 'LaMelo Ball',
        'age': 24,
        'games_played': 30,
        'games_remaining': 52,
        'season_avg_fp': 44.7,
        'last_5_avg_fp': 48.2,
        'last_10_avg_fp': 46.5,
        'last_15_avg_fp': 45.1,
        'fp_std_15': 9.8,
        'fp_std_30': 10.2,
        'mpg': 34.5,
        'usage_rate': 0.29,
        'ts_pct': 0.545,
        'per': 20.5,
        'games_missed_last_season': 35,
        'games_missed_current': 2,
        'injury_status': 'healthy',
        'contract_year': False,
        'years_until_free_agency': 2,
        'position': 'PG',
        'is_starter': True,
        'team_pace': 102.5,
        'team_win_pct': 0.450,
        'star_teammate_recently_left': False,
        'next_5_opponent_def_ratings': [110, 108, 112, 107, 109],
        'next_5_opponent_pace': [100, 102, 98, 101, 99],
        'historical_fp_vs_next_5_opponents': [42, 48, 39, 45, 46]
    }

    # Engineer features
    features = feature_engine.engineer_all_features(player_data)

    # Calculate projections
    high, low, expected = feature_engine.calculate_high_low_projections(features)

    print(f"\n{'=' * 60}")
    print(f"FANTASY PROJECTION: {player_data['player_name']}")
    print(f"{'=' * 60}")
    print(f"Current Season Avg: {player_data['season_avg_fp']:.1f} FP")
    print(f"Expected Average:   {expected:.1f} FP")
    print(f"High End (Ceiling): {high:.1f} FP")
    print(f"Low End (Floor):    {low:.1f} FP")
    print(f"\nKey Factors:")
    print(f"  - Injury Risk Score: {features['injury_risk_score']:.2f}")
    print(f"  - Schedule Difficulty: {features['schedule_difficulty']:.2f}")
    print(f"  - Breakout Probability: {features['breakout_probability']:.2f}")
    print(f"  - Contract Motivation: {features['contract_motivation_score']:.2f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    example_usage()

    print("\n" + "=" * 60)
    print("SYSTEM READY FOR TRAINING")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load your CSV/JSON data")
    print("2. Initialize model: model = FantasyBasketballCNN()")
    print("3. Build architecture: model.build_model()")
    print("4. Train: model.train(X_train, y_train, X_val, y_val)")
    print("5. Use TradeAnalyzer for trade recommendations")
    print("=" * 60)