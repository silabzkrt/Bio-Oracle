"""
Math Engine - State Extraction and Lotka-Volterra Population Dynamics
Calculates population metrics and predicts future states using differential equations
"""
import numpy as np
from scipy.integrate import odeint
from typing import List, Dict, Tuple
import logging


class StateExtractor:
    """
    Extracts population-level statistics from cell tracking data.
    Calculates density, growth rate, velocity statistics.
    """
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """
        Initialize state extractor.
        
        Args:
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
        self.logger = logging.getLogger(__name__)
        
        # History for calculating rates
        self.cell_counts_history = []
        self.timestamps_history = []
        self.density_history = []
        
    def calculate_population_density(self, cell_count: int) -> float:
        """
        Calculate population density: ρ = Cells / Area
        
        Args:
            cell_count: Number of cells detected
            
        Returns:
            Population density (cells per pixel)
        """
        density = cell_count / self.frame_area
        self.density_history.append(density)
        return density
    
    def calculate_growth_rate(self, current_count: int, timestamp: float) -> float:
        """
        Calculate growth rate: r = ΔCells / Δt
        
        Args:
            current_count: Current number of cells
            timestamp: Current timestamp in seconds
            
        Returns:
            Growth rate (cells per second)
        """
        self.cell_counts_history.append(current_count)
        self.timestamps_history.append(timestamp)
        
        # Need at least 2 data points
        if len(self.cell_counts_history) < 2:
            return 0.0
        
        # Calculate rate from recent history (last 5 samples)
        window_size = min(5, len(self.cell_counts_history))
        recent_counts = self.cell_counts_history[-window_size:]
        recent_times = self.timestamps_history[-window_size:]
        
        # Linear regression for smoothed rate
        delta_cells = recent_counts[-1] - recent_counts[0]
        delta_time = recent_times[-1] - recent_times[0]
        
        if delta_time == 0:
            return 0.0
        
        growth_rate = delta_cells / delta_time
        return growth_rate
    
    def calculate_per_capita_growth_rate(self, current_count: int, timestamp: float) -> float:
        """
        Calculate per-capita growth rate: r/N = (1/N)(dN/dt)
        
        Args:
            current_count: Current number of cells
            timestamp: Current timestamp
            
        Returns:
            Per-capita growth rate
        """
        growth_rate = self.calculate_growth_rate(current_count, timestamp)
        
        if current_count == 0:
            return 0.0
        
        return growth_rate / current_count
    
    def calculate_average_velocity(self, entities: List) -> Tuple[float, float]:
        """
        Calculate average velocity of all tracked entities.
        
        Args:
            entities: List of BioEntity objects
            
        Returns:
            Tuple of (average_speed, average_direction_degrees)
        """
        if not entities:
            return 0.0, 0.0
        
        speeds = []
        directions = []
        
        for entity in entities:
            if entity.velocity is not None:
                vx, vy = entity.velocity
                speed = np.sqrt(vx**2 + vy**2)
                direction = np.arctan2(vy, vx) * 180 / np.pi
                
                speeds.append(speed)
                directions.append(direction)
        
        if not speeds:
            return 0.0, 0.0
        
        avg_speed = np.mean(speeds)
        avg_direction = np.mean(directions)
        
        return avg_speed, avg_direction
    
    def get_population_state(self, cell_count: int, timestamp: float, 
                           entities: List = None) -> Dict:
        """
        Get complete population state.
        
        Args:
            cell_count: Current cell count
            timestamp: Current timestamp
            entities: List of tracked entities (optional)
            
        Returns:
            Dictionary with population metrics
        """
        density = self.calculate_population_density(cell_count)
        growth_rate = self.calculate_growth_rate(cell_count, timestamp)
        per_capita_rate = self.calculate_per_capita_growth_rate(cell_count, timestamp)
        
        state = {
            'cell_count': cell_count,
            'timestamp': timestamp,
            'density': density,
            'density_per_1000px': density * 1000,
            'growth_rate': growth_rate,
            'per_capita_growth_rate': per_capita_rate,
        }
        
        # Add velocity statistics if entities provided
        if entities:
            avg_speed, avg_direction = self.calculate_average_velocity(entities)
            state['avg_speed'] = avg_speed
            state['avg_direction'] = avg_direction
        
        return state
    
    def reset(self):
        """Reset all history."""
        self.cell_counts_history = []
        self.timestamps_history = []
        self.density_history = []


class LotkaVolterraSimulator:
    """
    Implements Lotka-Volterra equations for population dynamics.
    
    Single species logistic growth:
    dN/dt = r*N*(1 - N/K)
    
    Where:
    - N = population size
    - r = intrinsic growth rate
    - K = carrying capacity
    """
    
    def __init__(self, growth_rate: float = 0.1, carrying_capacity: float = 100.0):
        """
        Initialize Lotka-Volterra simulator.
        
        Args:
            growth_rate: Intrinsic growth rate (r)
            carrying_capacity: Carrying capacity (K) - max population
        """
        self.r = growth_rate
        self.K = carrying_capacity
        self.logger = logging.getLogger(__name__)
        
    def logistic_growth(self, N: float, t: float) -> float:
        """
        Logistic growth differential equation.
        dN/dt = r*N*(1 - N/K)
        
        Args:
            N: Current population
            t: Time (not used, but required by odeint)
            
        Returns:
            Rate of change dN/dt
        """
        return self.r * N * (1 - N / self.K)
    
    def predict_single_species(self, current_population: float, 
                              time_horizon: float = 10.0, 
                              time_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict population growth for single species.
        
        Args:
            current_population: Current cell count
            time_horizon: How many seconds to predict forward
            time_steps: Number of time points to calculate
            
        Returns:
            Tuple of (time_array, population_array)
        """
        # Time points
        t = np.linspace(0, time_horizon, time_steps)
        
        # Solve ODE
        N = odeint(self.logistic_growth, current_population, t)
        
        return t, N.flatten()
    
    def estimate_growth_rate_from_data(self, populations: List[float], 
                                      time_points: List[float]) -> float:
        """
        Estimate growth rate from historical data using linear regression.
        
        Args:
            populations: List of population counts
            time_points: Corresponding time points
            
        Returns:
            Estimated growth rate (r)
        """
        if len(populations) < 2:
            return self.r
        
        # Filter out zeros
        valid_data = [(t, N) for t, N in zip(time_points, populations) if N > 0]
        
        if len(valid_data) < 2:
            return self.r
        
        times = np.array([t for t, _ in valid_data])
        pops = np.array([N for _, N in valid_data])
        
        # Linear regression on log data
        log_pops = np.log(pops)
        
        if len(times) > 1:
            r_estimated = np.polyfit(times, log_pops, 1)[0]
            return r_estimated
        
        return self.r
    
    def adaptive_predict(self, current_population: float,
                        historical_populations: List[float],
                        historical_times: List[float],
                        time_horizon: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive prediction that estimates parameters from recent data.
        
        Args:
            current_population: Current cell count
            historical_populations: Recent population history
            historical_times: Corresponding time points
            time_horizon: Prediction horizon in seconds
            
        Returns:
            Tuple of (time_array, population_array)
        """
        # Estimate growth rate from recent data
        if len(historical_populations) >= 5:
            estimated_r = self.estimate_growth_rate_from_data(
                historical_populations[-10:],
                historical_times[-10:]
            )
            
            # Update growth rate (with bounds for stability)
            self.r = np.clip(estimated_r, -0.5, 0.5)
        
        # Predict using estimated parameters
        return self.predict_single_species(current_population, time_horizon)
    
    def get_prediction_at_time(self, current_population: float, 
                              target_time: float) -> float:
        """
        Get predicted population at specific time in future.
        
        Args:
            current_population: Current cell count
            target_time: Time in seconds to predict
            
        Returns:
            Predicted population count
        """
        t, N = self.predict_single_species(current_population, target_time, 10)
        return N[-1]


class PopulationOracle:
    """
    High-level interface combining state extraction and simulation.
    The "Oracle" that predicts future population states using Lotka-Volterra.
    """
    
    def __init__(self, initial_population: float = 50.0,
                 growth_rate: float = 0.1,
                 carrying_capacity: float = 100.0):
        """
        Initialize the Oracle.
        
        Args:
            initial_population: Starting population
            growth_rate: Growth rate parameter
            carrying_capacity: Maximum sustainable population
        """
        self.simulator = LotkaVolterraSimulator(growth_rate, carrying_capacity)
        self.logger = logging.getLogger(__name__)
        
        # History for adaptive predictions
        self.population_history = []
        self.time_history = []
        
    def update(self, current_population: float, timestamp: float):
        """
        Update the oracle with new observation.
        
        Args:
            current_population: Current observed population
            timestamp: Current time
        """
        self.population_history.append(current_population)
        self.time_history.append(timestamp)
        
        # Keep only recent history (last 100 samples)
        if len(self.population_history) > 100:
            self.population_history = self.population_history[-100:]
            self.time_history = self.time_history[-100:]
    
    def predict_future(self, prediction_seconds: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict population for next N seconds.
        
        Args:
            prediction_seconds: How many seconds to predict ahead
            
        Returns:
            Tuple of (time_array, predicted_population_array)
        """
        if not self.population_history:
            return np.array([0]), np.array([0])
        
        current_pop = self.population_history[-1]
        
        # Use adaptive prediction if enough history
        if len(self.population_history) >= 5:
            return self.simulator.adaptive_predict(
                current_pop,
                self.population_history,
                self.time_history,
                prediction_seconds
            )
        else:
            return self.simulator.predict_single_species(current_pop, prediction_seconds)
    
    def get_prediction_text(self, prediction_seconds: float = 10.0) -> str:
        """
        Get human-readable prediction text.
        
        Args:
            prediction_seconds: Prediction horizon
            
        Returns:
            Prediction description string
        """
        if len(self.population_history) < 2:
            return "Insufficient data"
        
        current_pop = self.population_history[-1]
        predicted_pop = self.simulator.get_prediction_at_time(current_pop, prediction_seconds)
        
        change = predicted_pop - current_pop
        change_percent = (change / current_pop * 100) if current_pop > 0 else 0
        
        return f"t+{prediction_seconds}s: {predicted_pop:.0f} cells ({change:+.0f}, {change_percent:+.1f}%)"
