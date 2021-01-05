import csv
import os
import pickle
from time import time

from gupb.controller.shallow_mind.arenna_wrapper import ArenaWrapper
from gupb.controller.shallow_mind.consts import StrategyAction, EPSILON, LEARNING_RATE, DISCOUNT_FACTOR, REWARD_CONST, \
    PUNISHMENT_CONST, LEARN
from gupb.controller.shallow_mind.q_learning import QLearning
from gupb.model import characters
from gupb.model.arenas import ArenaDescription
from gupb.model.characters import Action, ChampionKnowledge
from queue import SimpleQueue

path = os.path.join("./resources/models/shallow_mind", 'q_learning_new.pickle')
logs_path = os.path.join("./logs",
                         f'{time()}, EPS:{EPSILON}, LR:{LEARNING_RATE}, DF:{DISCOUNT_FACTOR}, RC:{REWARD_CONST}, PC:{PUNISHMENT_CONST}.csv')
logs = []


class ShallowMindController:
    def __init__(self, first_name: str):
        self.first_name: str = first_name
        self.arena: ArenaWrapper = None
        self.action_queue: SimpleQueue[Action] = SimpleQueue()
        self.q_learning = QLearning()
        try:
            with open(path, 'rb') as handle:
                d = pickle.load(handle)
                self.q_learning = QLearning(d)
        except Exception:
            self.q_learning = QLearning()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ShallowMindController):
            return self.first_name == other.first_name
        return False

    def __hash__(self) -> int:
        return hash(self.first_name)

    def save(self):
        if LEARN:
            with open(path, 'wb') as handle:
                pickle.dump(self.q_learning.q, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(logs_path, 'w') as handle:
                writer = csv.writer(handle, delimiter=",")
                writer.writerow(["reward", "arrived", "dist"])
                writer.writerows(logs)

    def reset(self, arena_description: ArenaDescription) -> None:
        reward = self.q_learning.reset(self.arena)
        if LEARN and self.arena:
            logs.append((reward, self.arena.position == self.arena.menhir_destination, self.arena.move_to_menhir.time))
        self.arena = ArenaWrapper(arena_description)

    def decide(self, knowledge: ChampionKnowledge) -> Action:
        self.arena.prepare_matrix(knowledge)
        action = self.q_learning.attempt(self.arena)
        if action == StrategyAction.GO_TO_MENHIR:
            return self.arena.move_to_menhir.action
        return self.arena.find_scan_action()

    @property
    def name(self) -> str:
        return f'ShallowMindController{self.first_name}'

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.GREY


POTENTIAL_CONTROLLERS = [
    ShallowMindController('test'),
]
