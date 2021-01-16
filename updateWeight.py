

from typing import overload
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, ClusterMixin


class UpdateWeight():

    def __init__(self,baseNodes,basevectors):

        self.baseNodes = baseNodes
        self.basevectors = basevectors
        self.num_signal = 0
        self.baseNum = len(baseNodes)
        self._reset_state()


    def _reset_state(self):
        self.nodeLocationIndex = np.array([], dtype=np.int32)
        self.winning_times = [0]*self.baseNum




    def update(self, inputNodes):
        self._reset_state()
        numberMark = 0
        for x in inputNodes:
            self.input_signal(x)
        return self



        # inputNode
    def input_signal(self, signal: np.ndarray):

        self.num_signal += 1

        winner, dists = self.__find_nearest_nodes(signal)

        self.__add_nodelocation(winner)

        self.__update_winner(winner[0], signal)




    def __add_nodelocation(self, winner):
        n = self.nodeLocationIndex.shape[0]
        self.nodeLocationIndex.resize(n + 1, 1, refcheck=False)
        self.nodeLocationIndex[-1, :] = winner[0]


    def __find_nearest_nodes(self, signal: np.ndarray):
        n = self.basevectors.shape[0]
        indexes = [0]
        sq_dists = [0.0]
        D = np.sum((self.basevectors - np.array([signal] * n))**2, 1)
        indexes[0] = np.nanargmin(D)
        sq_dists[0] = D[indexes[0]]
        D[indexes[0]] = float('nan')
        return indexes, sq_dists


    def __update_winner(self, winner_index, signal):
        self.winning_times[winner_index] += 1
        w = self.basevectors[winner_index]
        self.basevectors[winner_index] = w + (signal - w)/self.winning_times[winner_index]
