"""
Bio-Oracle Simulation Module
Implements Lotka-Volterra population dynamics models
Predicts future cell counts based on current population state
"""
import numpy as np
from scipy.integrate import odeint
from typing import Tuple, List
import logging


class LotkaVolterraSimulator:
    """
    Implements Lotka-Volterra equations for population dynamics.
    
    Single species logistic growth:
    dN/dt = r*N*(1 - N/K)
    
    Two species competition:
    dN1/dt = r1*N1*(1 - N1/K1 - α*N2/K1)
    dN2/dt = r2*N2*(1 - N2/K2 - β*N1/K2)
    """
    
    def __init__(self, growth_rate: float = 0.1, carrying_capacity: float = 100.0):
        """
        Initialize Lotka-Volterra simulator.
        
        Args:
            growth_rate: Intrinsic growth rate (r)
            carrying_capacity: Carrying capacity (K) - max population
        """
        self.r = growth_rate  # Intrinsic growth rate
        self.K = carrying_capacity  # Carrying capacity
        self.logger = logging.getLogger(__name__)
        
        # Competition coefficients (for two-species model)
        self.alpha = 0.5  # Effect of species 2 on species 1
        self.beta = 0.5   # Effect of species 1 on species 2
        
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
    
    def two_species_competition(self, state: List[float], t: float) -> List[float]:
        """
        Two-species competition model.
        
        dN1/dt = r1*N1*(1 - N1/K1 - α*N2/K1)
        dN2/dt = r2*N2*(1 - N2/K2 - β*N1/K2)
        
        Args:
            state: [N1, N2] population of both species
            t: Time
            
        Returns:
            [dN1/dt, dN2/dt]
        """
        N1, N2 = state
        
        # Species 1 parameters
        r1 = self.r
        K1 = self.K
        
        # Species 2 parameters (slightly different)
        r2 = self.r * 0.8  # Slightly lower growth rate
        K2 = self.K * 1.2  # Slightly higher carrying capacity
        
        # Competition equations
        dN1_dt = r1 * N1 * (1 - N1/K1 - self.alpha * N2/K1)
        dN2_dt = r2 * N2 * (1 - N2/K2 - self.beta * N1/K2)
        
        return [dN1_dt, dN2_dt]
    
    def predict_two_species(self, N1_current: float, N2_current: float,
                           time_horizon: float = 10.0,
                           time_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict population dynamics for two competing species.
        
        Args:
            N1_current: Current population of species 1
            N2_current: Current population of species 2
            time_horizon: Prediction time in seconds
            time_steps: Number of time points
            
        Returns:
            Tuple of (time_array, N1_array, N2_array)
        """
        # Initial state
        initial_state = [N1_current, N2_current]
        
        # Time points
        t = np.linspace(0, time_horizon, time_steps)
        
        # Solve ODEs
        solution = odeint(self.two_species_competition, initial_state, t)
        
        N1 = solution[:, 0]
        N2 = solution[:, 1]
        
        return t, N1, N2
    
    def estimate_growth_rate_from_data(self, populations: List[float], 
                                      time_points: List[float]) -> float:
        """
        Estimate growth rate from historical data.
        
        Args:
            populations: List of population counts
            time_points: Corresponding time points
            
        Returns:
            Estimated growth rate (r)
        """
        if len(populations) < 2:
            return self.r
        
        # Use linear regression on log-transformed data
        # ln(N) = ln(N0) + r*t  (for exponential phase)
        
        # Filter out zeros and calculate log
        valid_data = [(t, N) for t, N in zip(time_points, populations) if N > 0]
        
        if len(valid_data) < 2:
            return self.r
        
        times = np.array([t for t, _ in valid_data])
        pops = np.array([N for _, N in valid_data])
        
        # Simple linear regression on log data
        log_pops = np.log(pops)
        
        # Calculate slope (growth rate)
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
                historical_populations[-10:],  # Last 10 samples
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
        return N[-1]  # Return final predicted value


class PopulationOracle:
    """
    High-level interface combining state extraction and simulation.
    The "Oracle" that predicts future population states.
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
            return "Insufficient data for prediction"
        
        current_pop = self.population_history[-1]
        predicted_pop = self.simulator.get_prediction_at_time(current_pop, prediction_seconds)
        
        change = predicted_pop - current_pop
        change_percent = (change / current_pop * 100) if current_pop > 0 else 0
        
        return f"t+{prediction_seconds}s: {predicted_pop:.0f} cells ({change:+.0f}, {change_percent:+.1f}%)"
