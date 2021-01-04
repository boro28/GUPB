import math
from collections import defaultdict
from random import random, choice
from typing import Dict, Tuple, NamedTuple

import numpy as np

from gupb.controller.shallow_mind.arenna_wrapper import ArenaWrapper
from gupb.controller.shallow_mind.consts import REWARD_CONST, DISCOUNT_FACTOR, LEARNING_RATE, StrategyAction, EPSILON, \
    PUNISHMENT_CONST, MIST_BINS, DIST_PROPORTION_BINS
from gupb.controller.shallow_mind.utils import points_dist

State = NamedTuple('State', [('arena', str), ('menhir', int), ('mist', int)])


def default_value():
    return 0


class QLearning:
    def __init__(self, knowledge=None):
        if knowledge is None:
            knowledge = {}
        self.q: Dict[Tuple[State, StrategyAction], float] = defaultdict(default_value, knowledge)
        self.old_state: State = None
        self.old_action: StrategyAction = None
        self.reward_sum: float = 0.0
        print(len(list(self.q.values())))
        print(self.q)

    def reset(self, arena: ArenaWrapper) -> float:
        self.old_state = None
        self.old_action = None
        reward_sum = self.reward_sum
        self.reward_sum = 0.0
        return reward_sum

    def best_action(self, state: State) -> StrategyAction:
        actions = {action: self.q[(state, action)] for action in StrategyAction}
        # print(actions)
        return max(actions, key=actions.get)

    def pick_action(self, state: State) -> StrategyAction:
        if random() < EPSILON:
            return choice(list(StrategyAction))
        else:
            return self.best_action(state)

    def update_q(self, state: State, action: StrategyAction, reward: int) -> None:
        if self.old_state and self.old_action:
            self.q[(self.old_state, self.old_action)] += LEARNING_RATE * (
                    reward + DISCOUNT_FACTOR * self.q[(state, action)] - self.q[(self.old_state, self.old_action)])
        self.old_action = action
        self.old_state = state

    def attempt(self, arena: ArenaWrapper) -> StrategyAction:
        state = self.discretise(arena)
        action = self.pick_action(state)
        reward = self.calculate_reward(arena)
        # print({'position': arena.position, 'menhir': arena.menhir_position, 'reward': reward, state: state})
        self.reward_sum += reward
        self.update_q(state, action, reward)
        return action

    def discretise(self, arena: ArenaWrapper) -> State:
        mist_dist = arena.calc_mist_dist()
        menhir_dist = points_dist(arena.position, arena.menhir_destination)
        return State('mini',
                     np.digitize([arena.move_to_menhir.time / menhir_dist if menhir_dist else 0], DIST_PROPORTION_BINS)[
                         0],
                     np.digitize([mist_dist], MIST_BINS)[0])

    def calculate_reward(self, arena: ArenaWrapper) -> int:
        dist = arena.calc_mist_dist()
        if dist > 0:
            return math.floor(REWARD_CONST / dist)
        if arena.position == arena.menhir_destination:
            return 1
        elif dist == 0 or self.old_action == StrategyAction.GO_TO_MENHIR:
            return PUNISHMENT_CONST * -1
        else:
            return PUNISHMENT_CONST * dist
