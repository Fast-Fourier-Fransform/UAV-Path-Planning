import numpy as np


class BaseOptimizer:
    def __init__(self, nPop, MaxIt, VarMin, VarMax, dim, CostFunc):
        self.nPop = nPop
        self.MaxIt = MaxIt
        self.VarMin = np.array(VarMin)
        self.VarMax = np.array(VarMax)
        self.dim = dim
        self.CostFunc = CostFunc

        self.best_position = None
        self.best_score = float('inf')
        self.curve = np.zeros(MaxIt)

    def optimize(self):
        """子类必须实现此方法，返回 (best_position, best_score, curve)"""
        raise NotImplementedError("必须在子类中实现 optimize() 方法")