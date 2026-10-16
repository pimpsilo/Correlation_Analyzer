"""
Integration test for Löwdin Symmetric Orthogonalization in Correlation_Analyzer.
Tests PortfolioMonteCarloSimulator with both lowdin and cholesky methods.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from lowdin_orthogonalization import (
    compute_lowdin_matrices,
    compute_strategy_uniqueness,
    compute_lowdin_profile_distance
)
from multi_symphony_portfolio_monte_carlo import PortfolioMonteCarloSimulator


class TestLowdinIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate synthetic backtest returns for 3 symphonies
        np.random.seed(123)
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(250)]
        
        # Correlated returns
        r1 = np.random.normal(0.0008, 0.015, 250)
        r2 = 0.7 * r1 + np.random.normal(0.0005, 0.010, 250)
        r3 = -0.3 * r1 + np.random.normal(0.0003, 0.008, 250)
        
        cls.symphony_data = {
            "Tech_Momentum": pd.DataFrame({"date": dates, "returns": r1}),
            "Growth_Trend": pd.DataFrame({"date": dates, "returns": r2}),
            "Treasury_Hedge": pd.DataFrame({"date": dates, "returns": r3})
        }

    def test_lowdin_monte_carlo_execution(self):
        # Test Löwdin Monte Carlo simulator run
        sim_lowdin = PortfolioMonteCarloSimulator(
            symphony_data=self.symphony_data,
            initial_capital=50000,
            correlation_method="lowdin"
        )
        sim_lowdin.set_weights(method="equal")
        results = sim_lowdin.run_monte_carlo(n_simulations=100, n_days_forward=30)
        self.assertIsNotNone(results)
        self.assertEqual(sim_lowdin.correlation_method, "lowdin")

    def test_cholesky_monte_carlo_execution(self):
        # Test backward-compatible Cholesky run
        sim_cholesky = PortfolioMonteCarloSimulator(
            symphony_data=self.symphony_data,
            initial_capital=50000,
            correlation_method="cholesky"
        )
        sim_cholesky.set_weights(method="equal")
        results = sim_cholesky.run_monte_carlo(n_simulations=100, n_days_forward=30)
        self.assertIsNotNone(results)
        self.assertEqual(sim_cholesky.correlation_method, "cholesky")

    def test_uniqueness_and_profile_distance(self):
        corr = np.array([
            [1.0, 0.7, -0.3],
            [0.7, 1.0, -0.1],
            [-0.3, -0.1, 1.0]
        ])
        uniqueness = compute_strategy_uniqueness(corr)
        self.assertEqual(len(uniqueness), 3)
        # Treasury hedge (-0.3 correlation) should have higher uniqueness than Growth_Trend (0.7 with Tech)
        self.assertGreater(uniqueness[2], uniqueness[1])
        
        dist = compute_lowdin_profile_distance(corr)
        self.assertEqual(len(dist), 3)

if __name__ == "__main__":
    unittest.main()
