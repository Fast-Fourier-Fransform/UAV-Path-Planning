import numpy as np
from .base_algorithm import BaseOptimizer


class PSO(BaseOptimizer):
    def optimize(self):
        w, wdamp = 1.2, 0.99
        c1, c2 = 1.5, 1.5

        VelMax = 0.1 * (self.VarMax - self.VarMin)
        VelMin = -VelMax

        particles = []
        for i in range(self.nPop):
            pos = np.random.uniform(self.VarMin, self.VarMax, self.dim)
            cost = self.CostFunc(pos)
            particles.append({
                'Position': pos,
                'Velocity': np.zeros(self.dim),
                'Cost': cost,
                'Best_Position': pos.copy(),
                'Best_Cost': cost
            })
            if cost < self.best_score:
                self.best_score = cost
                self.best_position = pos.copy()

        for it in range(self.MaxIt):
            for i in range(self.nPop):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                particles[i]['Velocity'] = (w * particles[i]['Velocity'] +
                                            c1 * r1 * (particles[i]['Best_Position'] - particles[i]['Position']) +
                                            c2 * r2 * (self.best_position - particles[i]['Position']))

                particles[i]['Velocity'] = np.clip(particles[i]['Velocity'], VelMin, VelMax)
                particles[i]['Position'] += particles[i]['Velocity']

                # 越界处理：速度反弹，位置截断
                is_outside = (particles[i]['Position'] < self.VarMin) | (particles[i]['Position'] > self.VarMax)
                particles[i]['Velocity'][is_outside] = -particles[i]['Velocity'][is_outside]
                particles[i]['Position'] = np.clip(particles[i]['Position'], self.VarMin, self.VarMax)

                cost = self.CostFunc(particles[i]['Position'])
                particles[i]['Cost'] = cost

                if cost < particles[i]['Best_Cost']:
                    particles[i]['Best_Position'] = particles[i]['Position'].copy()
                    particles[i]['Best_Cost'] = cost
                    if cost < self.best_score:
                        self.best_score = cost
                        self.best_position = particles[i]['Position'].copy()

            self.curve[it] = self.best_score
            #print(f"Iteration {it + 1}: Best Cost = {self.best_score:.4f}")

        return self.best_position, self.best_score, self.curve