import gym
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy #Probalistic approch to also try some new paths from time to time
from rl.memory import SequentialMemory

from ale_py import ALEInterface, SDL_SUPPORT
import atari_py

ale = ALEInterface()
ale.setInt("random_seed", 123)

ale.loadROM(atari_py.get_game_path('pong'))


if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)

# Load the ROM file
#rom_file = sys.argv[0]
#ale.loadROM(rom_file)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        a = legal_actions[random.randrange(len(legal_actions))]
        # Apply an action and get the resulting reward
        reward = ale.act(a)
        total_reward += reward
    print("Episode %d ended with score: %d" % (episode, total_reward))
    ale.reset_game()