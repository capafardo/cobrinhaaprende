# ====================================================
# CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
# IBC - INSTITUTO BENJAMIN CONSTANT
# CURSO TÉCNICO NÍVEL MÉDIO EM INTELIGÊNCIA ARTIFICIAL
# Prof. Marcelo Fournier - nickname: capafardo
# ====================================================

import random
import numpy as np

# Ações Relativas ao sentido da cabeça
# 0: Continuar Reto
# 1: Virar à Esquerda
# 2: Virar à Direita
REL_ACTIONS = [0, 1, 2]
ACTION_NAMES = ["Continuar Reto", "Virar à Esquerda", "Virar à Direita"]

# Direções cardeais absolutas (para movimentação no grid)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.snake = [(cx, cy), (cx, cy + 1), (cx, cy + 2)]
        self.direction = UP
        self.score = 0
        self.steps_alive = 0
        self.steps_since_food = 0
        self.max_steps_without_food = 200
        self.alive = True
        self.food = self._spawn_food()

    def _spawn_food(self):
        empty_cells = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        if not empty_cells:
            return None
        return random.choice(empty_cells)

    def step(self, rel_action):
        """
        Executa um movimento com ação relativa:
        0: Manter a direção atual (Reto)
        1: Virar 90 graus à Esquerda
        2: Virar 90 graus à Direita
        """
        if not self.alive:
            return False

        dx, dy = self.direction

        if rel_action == 1:
            # Virar à Esquerda relativo à cabeça
            self.direction = (dy, -dx)
        elif rel_action == 2:
            # Virar à Direita relativo à cabeça
            self.direction = (-dy, dx)
        # rel_action == 0 mantém self.direction

        self.steps_alive += 1
        self.steps_since_food += 1

        head_x, head_y = self.snake[0]
        cur_dx, cur_dy = self.direction
        new_head = (head_x + cur_dx, head_y + cur_dy)

        # Checa colisão com parede
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.alive = False
            return False

        # Checa colisão com o próprio corpo
        if new_head in self.snake[:-1]:
            self.alive = False
            return False

        # Checa se morreu de fome (evita loop infinito sem maçã)
        if self.steps_since_food >= self.max_steps_without_food:
            self.alive = False
            return False

        # Move a cobra
        self.snake.insert(0, new_head)

        # Checa comida
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.max_steps_without_food = min(600, 200 + self.score * 50)
            self.food = self._spawn_food()
            if self.food is None:
                self.alive = False
                return True
        else:
            self.snake.pop()

        return True

    def get_vision(self):
        """
        Retorna 12 sensores de visão orientados ao ponto de vista da cabeça da cobra:
        1. Perigo imediato à Frente (1 passo de colisão com parede ou corpo)
        2. Perigo imediato à Esquerda (1 passo)
        3. Perigo imediato à Direita (1 passo)
        4. Distância normalizada de obstáculo à Frente (raio)
        5. Distância normalizada de obstáculo à Esquerda (raio)
        6. Distância normalizada de obstáculo à Direita (raio)
        7. Distância normalizada de obstáculo Diag Frente-Esquerda
        8. Distância normalizada de obstáculo Diag Frente-Direita
        9. Comida à Frente (1 se estiver no cone frontal, 0 caso contrário)
        10. Comida à Esquerda (1 se estiver à esquerda)
        11. Comida à Direita (1 se estiver à direita)
        12. Proximidade normalizada da Comida (1 / distância)
        """
        dx, dy = self.direction
        fwd = (dx, dy)
        lft = (dy, -dx)
        rgt = (-dy, dx)
        diag_fl = (dx + dy, dy - dx)
        diag_fr = (dx - dy, dy + dx)

        hx, hy = self.snake[0]

        def is_danger(pos):
            x, y = pos
            return (not (0 <= x < self.grid_size and 0 <= y < self.grid_size)) or (pos in self.snake[:-1])

        # 1-3. Perigo Imediato (a 1 passo da morte)
        p_fwd = 1.0 if is_danger((hx + fwd[0], hy + fwd[1])) else 0.0
        p_lft = 1.0 if is_danger((hx + lft[0], hy + lft[1])) else 0.0
        p_rgt = 1.0 if is_danger((hx + rgt[0], hy + rgt[1])) else 0.0

        # 4-8. Raios de Distância contínua até obstáculos (parede ou corpo)
        def dist_ray(dir_vec):
            d = 1
            cx, cy = hx + dir_vec[0], hy + dir_vec[1]
            while 0 <= cx < self.grid_size and 0 <= cy < self.grid_size and ((cx, cy) not in self.snake):
                cx += dir_vec[0]
                cy += dir_vec[1]
                d += 1
            return 1.0 / d

        dist_fwd = dist_ray(fwd)
        dist_lft = dist_ray(lft)
        dist_rgt = dist_ray(rgt)
        dist_dfl = dist_ray(diag_fl)
        dist_dfr = dist_ray(diag_fr)

        # 9-12. Posição Relativa da Comida
        if self.food:
            fx, fy = self.food
            f_vec = (fx - hx, fy - hy)
            food_fwd = 1.0 if (f_vec[0] * fwd[0] + f_vec[1] * fwd[1]) > 0 else 0.0
            food_lft = 1.0 if (f_vec[0] * lft[0] + f_vec[1] * lft[1]) > 0 else 0.0
            food_rgt = 1.0 if (f_vec[0] * rgt[0] + f_vec[1] * rgt[1]) > 0 else 0.0
            euclid_dist = np.sqrt(f_vec[0]**2 + f_vec[1]**2)
            food_proximity = 1.0 / (euclid_dist + 1e-4)
        else:
            food_fwd, food_lft, food_rgt, food_proximity = 0.0, 0.0, 0.0, 0.0

        return np.array([
            p_fwd, p_lft, p_rgt,
            dist_fwd, dist_lft, dist_rgt,
            dist_dfl, dist_dfr,
            food_fwd, food_lft, food_rgt,
            food_proximity
        ], dtype=np.float32)

    def calculate_fitness(self):
        """
        Função de aptidão otimizada para sobrevivência e coleta eficiente:
        Recompensa exponencial por maçãs e tempo de sobrevivência limpo.
        """
        apple_score = (self.score ** 2.2) * 500 + self.score * 250
        survival_score = self.steps_alive * 1.2
        fitness = apple_score + survival_score
        if self.score == 0:
            fitness = max(1.0, fitness * 0.3)
        return max(1.0, float(fitness))
