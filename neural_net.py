# ====================================================
# CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
# IBC - INSTITUTO BENJAMIN CONSTANT
# CURSO TÉCNICO NÍVEL MÉDIO EM INTELIGÊNCIA ARTIFICIAL
# Prof. Marcelo Fournier - nickname: capafardo
# ====================================================

import numpy as np

class NeuralNetwork:
    """
    Rede Neural Feedforward simples com 1 ou 2 camadas ocultas.
    Implementada diretamente em NumPy para permitir inspeção e evolução genética direta.
    """
    def __init__(self, input_size=24, hidden_sizes=(18, 14), output_size=4):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Inicialização dos pesos e biases (He / Xavier)
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        # Armazena ativações da última passagem para renderização didática
        self.last_activations = []

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, inputs):
        """
        Calcula as ativações camada por camada.
        inputs: array 1D ou 2D de tamanho input_size.
        Retorna as probabilidades/valores de saída e registra ativações para a GUI.
        """
        a = np.array(inputs, dtype=np.float32).reshape(1, -1)
        self.last_activations = [a[0].copy()]
        
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.last_activations.append(a[0].copy())
            
        # Camada final
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        out = self.softmax(z_out)
        self.last_activations.append(out[0].copy())
        return out[0]

    def clone(self):
        """Cria uma cópia exata da rede."""
        net = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        net.weights = [w.copy() for w in self.weights]
        net.biases = [b.copy() for b in self.biases]
        return net

    def mutate(self, mutation_rate=0.08, mutation_scale=0.25):
        """Aplica mutação gaussiana aleatória aos pesos e vieses."""
        for i in range(len(self.weights)):
            mask_w = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += mask_w * np.random.randn(*self.weights[i].shape) * mutation_scale
            
            mask_b = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.biases[i] += mask_b * np.random.randn(*self.biases[i].shape) * mutation_scale

    @staticmethod
    def crossover(parent1, parent2):
        """Gera um descendente combinando os pesos de dois pais."""
        child = parent1.clone()
        for i in range(len(child.weights)):
            # Crossover uniforme
            mask_w = np.random.rand(*child.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask_w, parent1.weights[i], parent2.weights[i])
            
            mask_b = np.random.rand(*child.biases[i].shape) > 0.5
            child.biases[i] = np.where(mask_b, parent1.biases[i], parent2.biases[i])
        return child
