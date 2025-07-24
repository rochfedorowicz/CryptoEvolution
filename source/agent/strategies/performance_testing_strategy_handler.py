# agent/strategies/performance_testing_strategy_handler.py

# global imports
import numpy as np
from typing import Any

# local imports
from source.agent import PerformanceTestable, TestingStrategyHandlerBase
from source.environment import TradingEnvironment

class PerformanceTestingStrategyHandler(TestingStrategyHandlerBase):
    """
    Implements a performance testing strategy handler for agents. It provides
    functionalities for evaluating the performance of agents in a trading environment.
    """

    # global class constants
    PLOTTING_KEY: str = 'performance_testing'

    def evaluate(self, testable_agent: PerformanceTestable, environment: TradingEnvironment) -> \
        tuple[list[str], list[dict[str, Any]]]:
        """
        Evaluates the performance of the given testable agent in the specified trading environment.

        Parameters:
            testable_agent (PerformanceTestable): The agent to evaluate.
            environment (TradingEnvironment): The trading environment to use for evaluation.

        Returns:
            (tuple[list[str], list[dict[str, Any]]]): A tuple containing the keys and data collected during evaluation.
        """

        history = {}
        assets_values = []
        reward_values = []
        infos = []
        iterations = []
        done = False

        state = environment.state
        current_iteration = environment.current_iteration
        trading_data = environment.get_trading_data()
        current_assets = trading_data.current_budget + trading_data.currently_invested
        iterations.append(current_iteration)
        assets_values.append(current_assets)
        reward_values.append(0)
        infos.append({})

        while(not done):
            next_action = testable_agent.perform(state)
            state, reward, done, info = environment.step(next_action)

            if current_assets != info['current_budget'] + info['currently_invested'] or done:
                current_iteration = environment.current_iteration
                current_assets = info['current_budget'] + info['currently_invested']
                iterations.append(current_iteration)
                assets_values.append(current_assets)
                reward_values.append(reward)
                infos.append(info)

        solvency_coefficient = round((assets_values[-1] - assets_values[0]) / (iterations[-1] - iterations[0]), 3)
        assets_values = (np.array(assets_values) / assets_values[0]).tolist()
        currency_prices = environment.get_data_for_iteration(['close'], iterations[0], iterations[-1])
        currency_prices = (np.array(currency_prices) / currency_prices[0]).tolist()

        history['assets_values'] = assets_values
        history['reward_values'] = reward_values
        history['currency_prices'] = currency_prices
        history['infos'] = infos
        history['iterations'] = iterations
        history['solvency_coefficient'] = solvency_coefficient

        return [self.PLOTTING_KEY], [history]