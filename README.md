```
====================================================
CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
IBC - INSTITUTO BENJAMIN CONSTANT
CURSO TÉCNICO NÍVEL MÉDIO EM INTELIGÊNCIA ARTIFICIAL
Prof. Marcelo Fournier - nickname: capafardo
====================================================
```

# 🐍 Cobrinha Aprende (Neuroevolução & Algoritmo Genético)

Uma aplicação interativa e educacional desenvolvida em **Python** que demonstra, visualmente e em tempo real, como uma **Rede Neural Artificial** pode aprender a jogar o clássico **Jogo da Cobrinha (Snake)** do absoluto zero, utilizando conceitos de **Algoritmos Genéticos (Neuroevolução)**.

Projetado especificamente para fins didáticos: qualquer pessoa — mesmo sem experiência prévia com Inteligência Artificial ou programação — consegue acompanhar o raciocínio da máquina, ver seus neurônios acendendo e entender como ocorre o processo de aprendizado.

---

## 🖥️ Como a Interface Funciona

A janela do programa é dividida didaticamente em duas áreas principais sincronizadas:

```
┌─────────────────────────────────┬──────────────────────────────────────────┐
│                                 │                                          │
│        1. O JOGO DA COBRINHA    │       2. O CÉREBRO DA COBRINHA           │
│     (Simulação Visual em 2D)    │        (Rede Neural em Tempo Real)       │
│                                 │                                          │
│  - Cobrinha navegando no grid   │  - 12 Sensores de entrada com rótulos    │
│  - Comida (maçã vermelha)       │  - 16 Neurônios ocultos piscando         │
│  - Raios coloridos de visão     │  - 3 Decisões de saída com porcentagens  │
│                                 │  - Sinapses coloridas (azul (+) / verm(-))│
├─────────────────────────────────┴──────────────────────────────────────────┤
│ 3. PAINEL DE CONTROLE E ESTATÍSTICAS                                       │
│ [Modo: Treino/JOGAR] [Pausar] [Velocidades: 1x, 5x, 20x, Turbo] [Salvar/Carregar]│
│ Indicador de Geração | Gráfico de Recordes | Alerta de Ponto Ideal de IA  │
└────────────────────────────────────────────────────────────────────────────┘
```

1. **Lado Esquerdo (O Jogo):** Mostra a melhor cobrinha da geração atual jogando, acompanhada de seus **raios sensoriais** (linhas que indicam o que ela está enxergando ao seu redor a cada instante).
2. **Lado Direito (A Rede Neural):** Um raio-X dinâmico do cérebro da cobrinha em funcionamento. Você vê os valores dos sensores entrando, as sinapses transmitindo impulsos elétricos e os neurônios acendendo em amarelo até a decisão final ser tomada.
3. **Painel Inferior (Controles e Métricas):** Permite acelerar a evolução, congelar o tempo para analisar uma decisão, salvar o melhor modelo ou alternar para o **Modo Play** (velocidade de jogo humano).

---

## 🧠 Como a Inteligência Artificial Aprende? (Em Termos Simples)

Muitas pessoas associam IA a bancos de dados gigantescos ou fórmulas indecifráveis. Neste projeto, a abordagem é inspirada diretamente na **Biologia e na Teoria da Evolução de Charles Darwin**:

### 1. A Rede Neural (O Sistema Nervoso)
A cobrinha possui um cérebro artificial formado por 3 camadas:
* **Entradas (Sensores - 12 Neurônios):** A cobra não enxerga a tela inteira como nós. Ela possui sensores relativos à sua cabeça:
  * *Perigo Imediato:* Sente se há parede ou o próprio corpo colado à sua **frente**, à sua **esquerda** ou à sua **direita**.
  * *Distâncias:* Mede a que distância está o obstáculo mais próximo em 5 direções.
  * *Bússola da Maçã:* Detecta se a comida está à sua frente, à esquerda ou à direita, além da distância geral.
* **Camada Oculta (Processamento - 16 Neurônios):** Combina as informações sensoriais (exemplo: *"tem comida à frente, MAS há uma parede colada na minha cara"*).
* **Saída (Decisões - 3 Neurônios):** A rede escolhe uma de três ações:
  1. `Continuar Reto`
  2. `Virar à Esquerda`
  3. `Virar à Direita`

As conexões entre os neurônios possuem pesos numéricos (**sinapses**):
* **Linhas Azuis (+):** Estímulos positivos (incentivam a ação).
* **Linhas Vermelhas (-):** Estímulos inibitórios (bloqueiam a ação).

---

### 2. O Algoritmo Genético (A Seleção Natural)
Como a cobra não tem um professor humano dizendo o que fazer, ela aprende por tentativa e erro:

1. **População Inicial (Geração 1):** Nascem 60 cobrinhas com cérebros completamente aleatórios. A maioria é desastrada e bate na parede nos primeiros segundos.
2. **Avaliação de Desempenho (*Fitness*):** Cada cobrinha recebe uma nota baseada em:
   * Quantas maçãs conseguiu comer (recompensa principal).
   * Quanto tempo conseguiu sobreviver sem bater (recompensa secundária).
   * Penalidade se ficar andando em círculos sem buscar comida.
3. **Seleção dos Melhores:** As cobras com as maiores notas são selecionadas como "pais".
4. **Reprodução (*Crossover*):** Combinamos os pesos do cérebro dos melhores indivíduos para gerar novos filhos.
5. **Mutação Genética:** Aplicamos pequenas alterações aleatórias nos pesos sinápticos dos filhos. Isso introduz novas ideias e comportamentos que os pais não tinham.
6. **Nova Geração:** O processo se repete. Em poucas gerações, surgem cobras que desviam perfeitamente das paredes e caçam as maçãs com alta precisão!

---

## 🎮 Funcionalidades e Controles

* **`Modo: Treino / JOGAR` (ou tecla `P`):**
  * *Treino:* A evolução corre solta para as cobras aprenderem.
  * *JOGAR (Modo Play):* Carrega automaticamente o modelo treinado e coloca a cobrinha para jogar em **velocidade normal de jogo humano** (~9 quadros por segundo), perfeito para apresentações em sala de aula!
* **`Pausar / Continuar` (ou tecla `Espaço`):** Congela o jogo e a rede para inspecionar com calma quais neurônios estão acesos naquele milissegundo.
* **Controles de Velocidade (`1x`, `5x`, `20x`, `Turbo`):** Permite treinar dezenas de gerações em poucos segundos no modo turbo.
* **`Raios: ON / OFF` (ou tecla `V`):** Liga ou desliga as linhas visuais dos sensores da cobrinha.
* **`Salvar Modelo`:** Grava os pesos da melhor cobra encontrada em um arquivo `melhor_cobra.pkl`.
* **`Carregar Modelo`:** Recarrega o arquivo salvo a qualquer momento.
* **`Reiniciar IA`:** Limpa a evolução e começa da Geração 1 novamente.
* **⭐ Alerta Automático de Maturidade:** O programa avisa na tela quando a cobrinha atinge um ponto ideal de aprendizado (platô de maestria), recomendando salvar o modelo.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
* **Python 3.10 ou superior** instalado no computador.

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/capafardo/cobrinha-aprende.git
cd cobrinha-aprende
```

2. **Crie e ative um ambiente virtual (recomendado):**
```bash
python3 -m venv .venv
source .venv/bin/activate   # No Linux/macOS
# No Windows use: .venv\Scripts\activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Inicie a aplicação:**
```bash
python3 main.py
```

---

## 📁 Estrutura dos Arquivos

```
cobrinha-aprende/
├── main.py                 # Interface gráfica principal em Pygame (duas telas, controles e gráficos)
├── snake_game.py           # O motor do jogo, grade 20x20 e os 12 sensores de visão da cobra
├── neural_net.py           # Implementação pura em NumPy da Rede Neural Feedforward
├── genetic_population.py   # Algoritmo genético: seleção por aptidão, crossover, mutação e elitismo
├── requirements.txt        # Dependências do projeto (pygame-ce, numpy, matplotlib, etc.)
├── README.md               # Documentação educacional completa do projeto
└── .gitignore              # Arquivos ignorados pelo controle de versão Git
```

---

## 👨‍🏫 Créditos e Instituição

* **Instituição:** CETAM - Centro de Educação Tecnológica do Amazonas
* **Unidade:** IBC - Instituto Benjamin Constant
* **Curso:** Curso Técnico de Nível Médio em Inteligência Artificial
* **Professor Responsável:** Prof. Marcelo Fournier (*nickname: capafardo*)
