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
from StockTradingEnv import StockTradingEnv

df = pd.read_csv("RL/EURUSD60.csv", delimiter='\t',
                 names=['Date','Open','High','Low','Close','Volume'])

# df = pd.read_csv('AAPL.csv')
# df = df.sort_values('Date')

MAX_ACCOUNT_BALANCE = 2147483647
INITIAL_ACCOUNT_BALANCE = 10000
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
obs = env.reset()

for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()