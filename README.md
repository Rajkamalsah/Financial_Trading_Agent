# Financial Trading Agent

This repository contains the implementation of a financial trading agent using reinforcement learning. The agent is designed to make buy/sell decisions based on historical stock data, utilizing a Deep Q-Network (DQN) to optimize its performance and profitability.

## Project Overview

The financial trading agent is developed using the following tools and libraries:
- **Python**
- **TensorFlow**
- **Keras**
- **Pandas**
- **Numpy**
- **Gym**

### Features
- **Reinforcement Learning**: The agent uses a DQN to learn and make trading decisions.
- **Historical Data**: The agent is trained on historical stock data to predict future price movements.
- **Performance Optimization**: Various strategies are implemented and tested to optimize the agent's performance.

## Installation

To install the required libraries, run the following command:
```bash
pip install numpy pandas tensorflow gym
```

## Usage

1. **Create the Environment**:
   ```python
   import gym
   import numpy as np
   import pandas as pd

   class TradingEnv(gym.Env):
       def __init__(self, df):
           super(TradingEnv, self).__init__()
           self.df = df
           self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
           self.observation_space = gym.spaces.Box(low=0, high=1, shape=(df.shape[1],), dtype=np.float32)
           self.current_step = 0
           self.balance = 10000
           self.shares_held = 0
           self.net_worth = 10000

       def reset(self):
           self.current_step = 0
           self.balance = 10000
           self.shares_held = 0
           self.net_worth = 10000
           return self._next_observation()

       def _next_observation(self):
           return self.df.iloc[self.current_step].values

       def step(self, action):
           current_price = self.df.iloc[self.current_step]['Close']
           if action == 1:  # Buy
               self.shares_held += self.balance // current_price
               self.balance %= current_price
           elif action == 2:  # Sell
               self.balance += self.shares_held * current_price
               self.shares_held = 0

           self.current_step += 1
           self.net_worth = self.balance + self.shares_held * current_price
           reward = self.net_worth - 10000
           done = self.current_step >= len(self.df) - 1
           return self._next_observation(), reward, done, {}

       def render(self, mode='human'):
           print(f'Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Net Worth: {self.net_worth}')
   ```

2. **Build the DQN Model**:
   ```python
   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Dropout

   def build_model(input_shape):
       model = Sequential()
       model.add(Dense(64, input_dim=input_shape, activation='relu'))
       model.add(Dropout(0.2))
       model.add(Dense(32, activation='relu'))
       model.add(Dense(3, activation='linear'))  # 3 actions: Hold, Buy, Sell
       model.compile(optimizer='adam', loss='mse')
       return model
   ```

3. **Train the Agent**:
   ```python
   from collections import deque
   import random

   class DQNAgent:
       def __init__(self, state_size, action_size):
           self.state_size = state_size
           self.action_size = action_size
           self.memory = deque(maxlen=2000)
           self.gamma = 0.95
           self.epsilon = 1.0
           self.epsilon_min = 0.01
           self.epsilon_decay = 0.995
           self.model = build_model(state_size)

       def remember(self, state, action, reward, next_state, done):
           self.memory.append((state, action, reward, next_state, done))

       def act(self, state):
           if np.random.rand() <= self.epsilon:
               return random.randrange(self.action_size)
           act_values = self.model.predict(state)
           return np.argmax(act_values[0])

       def replay(self, batch_size):
           minibatch = random.sample(self.memory, batch_size)
           for state, action, reward, next_state, done in minibatch:
               target = reward
               if not done:
                   target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
               target_f = self.model.predict(state)
               target_f[0][action] = target
               self.model.fit(state, target_f, epochs=1, verbose=0)
           if self.epsilon > self.epsilon_min:
               self.epsilon *= self.epsilon_decay

   # Load your data
   df = pd.read_csv('your_stock_data.csv')
   env = TradingEnv(df)
   state_size = env.observation_space.shape[0]
   action_size = env.action_space.n
   agent = DQNAgent(state_size, action_size)

   episodes = 1000
   batch_size = 32

   for e in range(episodes):
       state = env.reset()
       state = np.reshape(state, [1, state_size])
       for time in range(500):
           action = agent.act(state)
           next_state, reward, done, _ = env.step(action)
           reward = reward if not done else -10
           next_state = np.reshape(next_state, [1, state_size])
           agent.remember(state, action, reward, next_state, done)
           state = next_state
           if done:
               print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
               break
           if len(agent.memory) > batch_size:
               agent.replay(batch_size)
   ```

4. **Save the Model**:
   ```python
   agent.model.save('financial_trading_agent.h5')
   ```

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
