import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing
import json

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from datetime import datetime
import random

MAX_ACCOUNT_BALANCE = 2147483647
INITIAL_ACCOUNT_BALANCE = 10000
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.length_data = 10
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.lot = 0.1
        self.gain = 10
        self.loss = 20

        # Save Open trades in a list with indexes of how many open trades
        self.Open_trade = []
        self.number_open_trades = 0

        # Actions of the format Buy curr, Sell curr, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.pos_opened = []
        self.number_open_pos = 0
        self.cost_basis = 0
        self.positions_closed = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - self.length_data - 1)
        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                                           self.length_data, 'Open'].values,
            self.df.loc[self.current_step: self.current_step +
                                           self.length_data, 'High'].values,
            self.df.loc[self.current_step: self.current_step +
                                           self.length_data, 'Low'].values,
            self.df.loc[self.current_step: self.current_step +
                                           self.length_data, 'Close'].values,
            self.df.loc[self.current_step: self.current_step +
                                           self.length_data, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / MAX_NUM_SHARES,
        ]], axis=0)

        return obs

    def closing_position(self):
        i = 0
        while i < self.number_open_trades:
            if self.Open_trade[i]['position'] == 1:
                # Check first if the min is below the SL just to be sure that during the movement of the candle the SL was
                # not touched
                if self.row['Low'] <= self.Open_trade[i]['SL']:
                    self.gains += (self.Open_trade[i]['SL'] - self.Open_trade[i]['entry']) * 10000
                    # print('Position: BUY \n' + 'Open:  ' + str(self.Open_trade[i]['entry']) + '\nClose: ' + str(
                    #    self.row['Low']) + '\nProfit: ' + str(
                    #    (self.Open_trade[i]['SL'] - self.Open_trade[i]['entry']) * 10000) + '\n Gains: ' + str(self.gains) + '\n')
                    if (self.Open_trade[i]['SL'] - self.Open_trade[i]['entry']) > 0:
                        self.Pos_counter += 1
                    else:
                        self.Neg_counter += 1
                    del self.Open_trade[i]
                    self.number_open_trades -= 1
                    self.Open_counter -= 1
                    self.Closed_counter += 1

                elif self.row['Close'] <= self.Open_trade[i]['SL']:
                    self.gains += (self.Open_trade[i]['SL'] - self.Open_trade[i]['entry']) * 10000
                    # print('Position: BUY \n' + 'Open:  ' + str(self.Open_trade[i]['entry']) + '\nClose: ' + str(
                    #     self.row['Close']) + '\nProfit: ' + str(
                    #     (self.Open_trade[i]['SL'] - self.Open_trade[i]['entry']) * 10000) + '\n Gains: ' + str(self.gains) + '\n')
                    if (self.Open_trade[i]['SL'] - self.Open_trade[i]['entry']) > 0:
                        self.Pos_counter += 1
                    else:
                        self.Neg_counter += 1
                    del self.Open_trade[i]
                    self.number_open_trades -= 1
                    self.Open_counter -= 1
                    self.Closed_counter += 1

                elif self.row['Close'] >= self.Open_trade[i]['TP']:
                    self.gains += (self.Open_trade[i]['TP'] - self.Open_trade[i]['entry']) * 10000
                    # print('Position: BUY \n' + 'Open:  ' + str(self.Open_trade[i]['entry']) + '\nClose: ' + str(
                    #     self.Open_trade[i]['TP']) + '\nProfit: ' + str(
                    #     (self.Open_trade[i]['TP'] - self.Open_trade[i]['entry']) * 10000) + '\n Gains: ' + str(self.gains) + '\n')
                    if (self.Open_trade[i]['TP'] - self.Open_trade[i]['entry']) > 0:
                        self.Pos_counter += 1
                    else:
                        self.Neg_counter += 1
                    del self.Open_trade[i]
                    self.number_open_trades -= 1
                    self.Open_counter -= 1
                    self.Closed_counter += 1

            elif self.Open_trade[i]['position'] == -1:
                if self.row['High'] >= self.Open_trade[i]['SL']:
                    self.gains += (-self.Open_trade[i]['SL'] + self.Open_trade[i]['entry']) * 10000
                    # print('Position: SELL \n' + 'Open:  ' + str(self.Open_trade[i]['entry']) + '\nClose: ' + str(
                    #     self.row['High']) + '\nProfit: ' + str(
                    #     (-self.Open_trade[i]['SL'] + self.Open_trade[i]['entry']) * 10000) + '\n Gains: ' + str(self.gains) + '\n')
                    if (-self.Open_trade[i]['SL'] + self.Open_trade[i]['entry']) > 0:
                        self.Pos_counter += 1
                    else:
                        self.Neg_counter += 1
                    del self.Open_trade[i]
                    self.number_open_trades -= 1
                    self.Open_counter -= 1
                    self.Closed_counter += 1

                elif self.row['Close'] >= self.Open_trade[i]['SL']:
                    self.gains += (-self.Open_trade[i]['SL'] + self.Open_trade[i]['entry']) * 10000
                    # print('Position: SELL \n' + 'Open:  ' + str(self.Open_trade[i]['entry']) + '\n Close: ' + str(
                    #     self.row['Close']) + '\n Profit: ' + str(
                    #     (-self.Open_trade[i]['SL'] + self.Open_trade[i]['entry']) * 10000) + '\n Gains: ' + str(self.gains) + '\n')
                    if (-self.Open_trade[i]['SL'] + self.Open_trade[i]['entry']) > 0:
                        self.Pos_counter += 1
                    else:
                        self.Neg_counter += 1
                    del self.Open_trade[i]
                    self.number_open_trades -= 1
                    self.Open_counter -= 1
                    self.Closed_counter += 1

                elif self.row['Close'] <= self.Open_trade[i]['TP']:
                    self.gains += (-self.Open_trade[i]['TP'] + self.Open_trade[i]['entry']) * 10000
                    # print('Position: SELL \n' + 'Open:  ' + str(self.Open_trade[i]['entry']) + '\n Close: ' + str(
                    #     self.Open_trade[i]['TP']) + '\n Profit: ' + str(
                    #     (-self.Open_trade[i]['TP'] + self.Open_trade[i]['entry']) * 10000) + '\n Gains: ' + str(self.gains) + '\n')
                    if (-self.Open_trade[i]['TP'] + self.Open_trade[i]['entry']) > 0:
                        self.Pos_counter += 1
                    else:
                        self.Neg_counter += 1
                    del self.Open_trade[i]
                    self.number_open_trades -= 1
                    self.Open_counter -= 1
                    self.Closed_counter += 1
            i += 1

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, "Open"],
                                       self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            trade = {'entry': current_price, 'position': action_type, 'lot': self.lot,
                     'SL': current_price - self.loss / 10000,
                     'TP': current_price + self.gain / 10000, 'index': self.number_open_pos}

            self.number_open_trades += 1
            self.Open_counter += 1

            self.pos_opened.append(trade)

        elif action_type < 2:
            trade = {'entry': current_price, 'position': action_type, 'lot': self.lot,
                     'SL': current_price + self.loss / 10000,
                     'TP': current_price - self.gain / 10000, 'index': self.number_open_pos}
            self.number_open_trades += 1
            self.Open_counter += 1
            self.pos_opened.append(trade)

        elif action_type < 3:
            i = 0
            while i < self.number_open_trades:
                self.gains += (current_price - self.Open_trade[i]['entry']) * 10000
                if (current_price - self.Open_trade[i]['entry']) > 0:
                    self.Pos_counter += 1
                else:
                    self.Neg_counter += 1
                del self.Open_trade[i]
                self.number_open_trades -= 1
                self.Open_counter -= 1
                self.Closed_counter += 1

        self.netWorth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')