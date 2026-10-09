# ====================================================
# CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
# IBC - INSTITUTO BENJAMIN CONSTANT
# CURSO TÉCNICO NÍVEL MÉDIO EM INTELIGÊNCIA ARTIFICIAL
# Prof. Marcelo Fournier - nickname: capafardo
# ====================================================

import pickle
import numpy as np
from neural_net import NeuralNetwork
from snake_game import SnakeGame

class SnakeAgent:
    def __init__(self, brain=None):
        # 12 entradas sensoriais relativas, 1 camada oculta de 16 neurônios, 3 saídas (Reto, Esquerda, Direita)
        self.brain = brain if brain is not None else NeuralNetwork(input_size=12, hidden_sizes=(16,), output_size=3)
        self.game = SnakeGame()
        self.fitness = 0.0
        self.is_alive = True

    def step(self):
        if not self.is_alive:
            return False
        vision = self.game.get_vision()
        outputs = self.brain.forward(vision)
        action_idx = int(np.argmax(outputs))
        moved = self.game.step(action_idx)
        if not moved or not self.game.alive:
            self.is_alive = False
            self.fitness = self.game.calculate_fitness()
            return False
        return True

    def reset_game(self):
        self.game.reset()
        self.is_alive = True
        self.fitness = 0.0

class Population:
    def __init__(self, size=60, mutation_rate=0.10, mutation_scale=0.25):
        self.size = size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.generation = 1
        self.agents = [SnakeAgent() for _ in range(size)]
        self.best_agent_ever = None
        self.best_score_ever = 0
        self.best_fitness_ever = 0
        self.history_best_scores = []
        self.history_avg_scores = []
        self.history_best_fitness = []

        # Auto-verificação de maturidade
        self.stagnant_generations = 0
        self.recommended_stop = False
        self.stop_reason = ""

    def get_alive_count(self):
        return sum(1 for a in self.agents if a.is_alive)

    def get_leading_agent(self):
        alive_agents = [a for a in self.agents if a.is_alive]
        if alive_agents:
            return max(alive_agents, key=lambda a: (a.game.score, a.game.steps_alive))
        return max(self.agents, key=lambda a: a.fitness)

    def step_all(self):
        any_alive = False
        for a in self.agents:
            if a.is_alive:
                if a.step():
                    any_alive = True
        return any_alive

    def check_learning_maturity(self, gen_best_score, gen_avg_score):
        """
        Auto-verificação:
        Avisa o usuário quando a rede atingiu um nível de maestria (ex: >= 25 maçãs)
        ou se atingiu platô estável consolidado.
        """
        if self.best_score_ever >= 30:
            self.recommended_stop = True
            self.stop_reason = f"Desempenho excelente! Cobrinha alcançou {self.best_score_ever} maçãs com domínio do espaço."
            return

        if self.generation >= 25 and self.best_score_ever >= 15:
            if self.stagnant_generations >= 15:
                self.recommended_stop = True
                self.stop_reason = f"Convergência atingida ({self.best_score_ever} maçãs estáveis). Pronto para jogar!"
                return

    def evolve(self):
        for a in self.agents:
            if a.fitness == 0:
                a.fitness = a.game.calculate_fitness()

        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        gen_best = self.agents[0]
        gen_best_score = max(a.game.score for a in self.agents)
        gen_avg_score = sum(a.game.score for a in self.agents) / self.size
        
        self.history_best_scores.append(gen_best_score)
        self.history_avg_scores.append(gen_avg_score)
        self.history_best_fitness.append(gen_best.fitness)

        improved = False
        if self.best_agent_ever is None or gen_best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = gen_best.fitness
            if gen_best_score > self.best_score_ever:
                self.best_score_ever = gen_best_score
                improved = True
            self.best_agent_ever = SnakeAgent(brain=gen_best.brain.clone())

        if improved:
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1

        self.check_learning_maturity(gen_best_score, gen_avg_score)

        # Mutação Adaptativa
        cur_mut_rate = self.mutation_rate
        cur_mut_scale = self.mutation_scale
        if self.stagnant_generations > 10:
            cur_mut_rate = min(0.22, self.mutation_rate * 1.5)
            cur_mut_scale = min(0.35, self.mutation_scale * 1.4)

        # Elitismo: 4 melhores passam direto
        new_agents = []
        elite_count = min(4, self.size)
        for i in range(elite_count):
            elite = SnakeAgent(brain=self.agents[i].brain.clone())
            new_agents.append(elite)

        # Seleção por Roleta e Reprodução
        fitnesses = np.array([max(0.1, a.fitness) for a in self.agents])
        probs = fitnesses / np.sum(fitnesses)

        while len(new_agents) < self.size:
            p1_idx = np.random.choice(self.size, p=probs)
            p2_idx = np.random.choice(self.size, p=probs)
            p1 = self.agents[p1_idx]
            p2 = self.agents[p2_idx]

            child_brain = NeuralNetwork.crossover(p1.brain, p2.brain)
            child_brain.mutate(cur_mut_rate, cur_mut_scale)
            new_agents.append(SnakeAgent(brain=child_brain))

        self.agents = new_agents
        self.generation += 1

    def save_best(self, filepath="melhor_cobra.pkl"):
        target = self.best_agent_ever if self.best_agent_ever else self.agents[0]
        data = {
            "weights": target.brain.weights,
            "biases": target.brain.biases,
            "hidden_sizes": target.brain.hidden_sizes,
            "input_size": target.brain.input_size,
            "output_size": target.brain.output_size,
            "generation": self.generation,
            "best_score": self.best_score_ever
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_best(self, filepath="melhor_cobra.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        brain = NeuralNetwork(data["input_size"], data["hidden_sizes"], data["output_size"])
        brain.weights = data["weights"]
        brain.biases = data["biases"]
        self.best_agent_ever = SnakeAgent(brain=brain)
        self.agents = [SnakeAgent(brain=brain.clone())]
        for _ in range(self.size - 1):
            clone_b = brain.clone()
            clone_b.mutate(self.mutation_rate, self.mutation_scale)
            self.agents.append(SnakeAgent(brain=clone_b))
        self.generation = data.get("generation", 1)
        self.best_score_ever = data.get("best_score", 0)
        self.stagnant_generations = 0
        self.recommended_stop = False
