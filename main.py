# ====================================================
# CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
# IBC - INSTITUTO BENJAMIN CONSTANT
# CURSO TÉCNICO NÍVEL MÉDIO EM INTELIGÊNCIA ARTIFICIAL
# Prof. Marcelo Fournier - nickname: capafardo
# ====================================================

import sys
import os
import pygame
import numpy as np
from genetic_population import Population, SnakeAgent
from neural_net import NeuralNetwork
from snake_game import ACTION_NAMES

# Inicializa Pygame
pygame.init()
pygame.font.init()

# Cores
COLOR_BG = (16, 17, 24)
COLOR_PANEL_BG = (25, 27, 38)
COLOR_GRID = (35, 38, 54)
COLOR_SNAKE_HEAD = (80, 220, 100)
COLOR_SNAKE_BODY = (46, 160, 67)
COLOR_SNAKE_DEAD = (140, 60, 60)
COLOR_FOOD = (255, 75, 75)
COLOR_TEXT_MAIN = (240, 240, 245)
COLOR_TEXT_MUTED = (160, 165, 180)
COLOR_ACCENT = (70, 130, 250)
COLOR_ACCENT_HOVER = (95, 150, 255)
COLOR_WEIGHT_POS = (66, 135, 245)  # Azul para sinapse positiva
COLOR_WEIGHT_NEG = (245, 80, 80)   # Vermelho para sinapse negativa
COLOR_NODE_ACTIVE = (255, 215, 0)  # Amarelo brilhante quando ativado
COLOR_NODE_INACTIVE = (55, 60, 80)
COLOR_ALERT_BG = (220, 150, 20)
COLOR_PLAY_MODE = (40, 180, 120)

# Dimensões da Janela
WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 770

# Áreas
GAME_AREA_X = 20
GAME_AREA_Y = 20
GAME_SIZE = 560
GRID_CELL_SIZE = GAME_SIZE // 20

NEURAL_AREA_X = 600
NEURAL_AREA_Y = 20
NEURAL_AREA_WIDTH = 730
NEURAL_AREA_HEIGHT = 500

CONTROLS_AREA_X = 20
CONTROLS_AREA_Y = 600
CONTROLS_AREA_WIDTH = 1310
CONTROLS_AREA_HEIGHT = 145

# Nomes didáticos dos sensores de entrada (12)
INPUT_LABELS = [
    "Perigo Imediato Frente",
    "Perigo Imediato Esquerda",
    "Perigo Imediato Direita",
    "Obstáculo Dist Frente",
    "Obstáculo Dist Esquerda",
    "Obstáculo Dist Direita",
    "Obstáculo Diag Esq",
    "Obstáculo Diag Dir",
    "Comida à Frente",
    "Comida à Esquerda",
    "Comida à Direita",
    "Proximidade Comida"
]

# Fontes
FONT_TITLE = pygame.font.SysFont("DejaVu Sans, Ubuntu, Arial", 17, bold=True)
FONT_BODY = pygame.font.SysFont("DejaVu Sans, Ubuntu, Arial", 13)
FONT_SMALL = pygame.font.SysFont("DejaVu Sans, Ubuntu, Arial", 11)
FONT_TINY = pygame.font.SysFont("DejaVu Sans, Ubuntu, Arial", 9)

class CobrinhaEducacionalGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Cobrinha com Rede Neural & Algoritmo Genético - Aprendizado Visual")
        self.clock = pygame.time.Clock()
        self.running = True

        self.population = Population(size=60)
        self.speed_multiplier = 1
        self.paused = False
        self.show_vision_rays = True
        self.status_message = "Treinamento iniciado: Geração 1"

        # Modo Play
        self.play_mode = False
        self.play_agent = None
        self.play_game_over_timer = 0
        self.play_fps = 9

        # Botões
        self.buttons = [
            {"id": "mode_toggle", "text": "Modo: Treino", "rect": pygame.Rect(30, 642, 125, 36)},
            {"id": "pause", "text": "Pausar", "rect": pygame.Rect(165, 642, 95, 36)},
            {"id": "speed_1", "text": "1x", "rect": pygame.Rect(270, 642, 45, 36)},
            {"id": "speed_5", "text": "5x", "rect": pygame.Rect(325, 642, 45, 36)},
            {"id": "speed_20", "text": "20x", "rect": pygame.Rect(380, 642, 50, 36)},
            {"id": "speed_turbo", "text": "Turbo", "rect": pygame.Rect(440, 642, 65, 36)},
            {"id": "toggle_vision", "text": "Raios: ON", "rect": pygame.Rect(515, 642, 95, 36)},
            {"id": "save", "text": "Salvar Modelo", "rect": pygame.Rect(620, 642, 120, 36)},
            {"id": "load", "text": "Carregar Modelo", "rect": pygame.Rect(750, 642, 135, 36)},
            {"id": "restart", "text": "Reiniciar IA", "rect": pygame.Rect(895, 642, 110, 36)},
        ]

    def run(self):
        while self.running:
            self.handle_events()

            if not self.paused:
                if self.play_mode:
                    self.update_play_mode()
                else:
                    steps_per_frame = self.speed_multiplier
                    for _ in range(steps_per_frame):
                        any_alive = self.population.step_all()
                        if not any_alive:
                            self.population.evolve()
                            self.status_message = f"Geração {self.population.generation} | Recorde: {self.population.best_score_ever} maçãs."
                            break

            self.render()
            
            if self.play_mode:
                self.clock.tick(self.play_fps)
            else:
                self.clock.tick(60 if self.speed_multiplier <= 20 else 120)

        pygame.quit()
        sys.exit()

    def update_play_mode(self):
        if not self.play_agent:
            self.load_for_play()
            return

        if self.play_agent.is_alive:
            self.play_agent.step()
        else:
            self.play_game_over_timer += 1
            if self.play_game_over_timer > 15:
                self.play_agent.reset_game()
                self.play_game_over_timer = 0

    def load_for_play(self):
        if os.path.exists("melhor_cobra.pkl"):
            import pickle
            with open("melhor_cobra.pkl", "rb") as f:
                data = pickle.load(f)
            brain = NeuralNetwork(data["input_size"], data["hidden_sizes"], data["output_size"])
            brain.weights = data["weights"]
            brain.biases = data["biases"]
            self.play_agent = SnakeAgent(brain=brain)
            self.status_message = f"Modo Play ativo com modelo salvo (Recorde: {data.get('best_score', 0)} maçãs)!"
        else:
            best = self.population.best_agent_ever if self.population.best_agent_ever else self.population.agents[0]
            self.play_agent = SnakeAgent(brain=best.brain.clone())
            self.status_message = "Modo Play ativo usando a melhor cobra da sessão atual!"

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                self.handle_click(mouse_pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.update_pause_btn_text()
                elif event.key == pygame.K_v:
                    self.show_vision_rays = not self.show_vision_rays
                elif event.key == pygame.K_p:
                    self.toggle_play_mode()

    def update_pause_btn_text(self):
        for b in self.buttons:
            if b["id"] == "pause":
                b["text"] = "Continuar" if self.paused else "Pausar"

    def toggle_play_mode(self):
        self.play_mode = not self.play_mode
        for b in self.buttons:
            if b["id"] == "mode_toggle":
                b["text"] = "Modo: JOGAR" if self.play_mode else "Modo: Treino"
        if self.play_mode:
            self.load_for_play()
        else:
            self.status_message = f"Retomando treino na Geração {self.population.generation}."

    def handle_click(self, mouse_pos):
        for btn in self.buttons:
            if btn["rect"].collidepoint(mouse_pos):
                btn_id = btn["id"]
                if btn_id == "mode_toggle":
                    self.toggle_play_mode()
                elif btn_id == "pause":
                    self.paused = not self.paused
                    self.update_pause_btn_text()
                elif btn_id == "speed_1":
                    self.speed_multiplier = 1
                elif btn_id == "speed_5":
                    self.speed_multiplier = 5
                elif btn_id == "speed_20":
                    self.speed_multiplier = 20
                elif btn_id == "speed_turbo":
                    self.speed_multiplier = 100
                elif btn_id == "toggle_vision":
                    self.show_vision_rays = not self.show_vision_rays
                    btn["text"] = f"Raios: {'ON' if self.show_vision_rays else 'OFF'}"
                elif btn_id == "save":
                    self.population.save_best("melhor_cobra.pkl")
                    self.status_message = "Melhor modelo salvo em 'melhor_cobra.pkl'!"
                elif btn_id == "load":
                    if os.path.exists("melhor_cobra.pkl"):
                        self.population.load_best("melhor_cobra.pkl")
                        if self.play_mode:
                            self.load_for_play()
                        self.status_message = "Modelo carregado com sucesso!"
                    else:
                        self.status_message = "Arquivo 'melhor_cobra.pkl' não encontrado!"
                elif btn_id == "restart":
                    self.population = Population(size=60)
                    self.play_agent = None
                    self.status_message = "Evolução reiniciada do zero."

    def get_current_active_agent(self):
        if self.play_mode and self.play_agent:
            return self.play_agent
        return self.population.get_leading_agent()

    def render(self):
        self.screen.fill(COLOR_BG)
        self.render_game_area()
        self.render_neural_panel()
        self.render_controls_and_stats()
        pygame.display.flip()

    def render_game_area(self):
        game_rect = pygame.Rect(GAME_AREA_X, GAME_AREA_Y, GAME_SIZE, GAME_SIZE)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, game_rect, border_radius=10)
        border_col = COLOR_PLAY_MODE if self.play_mode else COLOR_GRID
        pygame.draw.rect(self.screen, border_col, game_rect, width=2, border_radius=10)

        # Grade
        for i in range(21):
            x = GAME_AREA_X + i * GRID_CELL_SIZE
            pygame.draw.line(self.screen, (32, 35, 48), (x, GAME_AREA_Y), (x, GAME_AREA_Y + GAME_SIZE))
            y = GAME_AREA_Y + i * GRID_CELL_SIZE
            pygame.draw.line(self.screen, (32, 35, 48), (GAME_AREA_X, y), (GAME_AREA_X + GAME_SIZE, y))

        active_agent = self.get_current_active_agent()
        game = active_agent.game

        # Raios de visão relativos à cabeça
        if self.show_vision_rays and game.alive:
            hx, hy = game.snake[0]
            center_head_x = GAME_AREA_X + hx * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            center_head_y = GAME_AREA_Y + hy * GRID_CELL_SIZE + GRID_CELL_SIZE // 2

            dx, dy = game.direction
            # 5 direções de visão da cabeça: Frente, Esquerda, Direita, Diag Esq, Diag Dir
            rel_rays = [
                ((dx, dy), (255, 100, 100)),        # Frente
                ((dy, -dx), (80, 180, 255)),       # Esquerda
                ((-dy, dx), (80, 180, 255)),       # Direita
                ((dx + dy, dy - dx), (200, 100, 255)), # Diag Esq
                ((dx - dy, dy + dx), (200, 100, 255)), # Diag Dir
            ]

            for (rx, ry), ray_color in rel_rays:
                cur_x, cur_y = hx + rx, hy + ry
                ray_target = None
                while 0 <= cur_x < 20 and 0 <= cur_y < 20:
                    if (cur_x, cur_y) in game.snake:
                        ray_target = (cur_x, cur_y)
                        break
                    cur_x += rx
                    cur_y += ry
                
                if ray_target is None:
                    cur_x = max(0, min(19, cur_x))
                    cur_y = max(0, min(19, cur_y))
                    ray_target = (cur_x, cur_y)

                tx = GAME_AREA_X + ray_target[0] * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
                ty = GAME_AREA_Y + ray_target[1] * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
                pygame.draw.line(self.screen, ray_color, (center_head_x, center_head_y), (tx, ty), 1)

            # Linha direta até a comida
            if game.food:
                fx, fy = game.food
                cfx = GAME_AREA_X + fx * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
                cfy = GAME_AREA_Y + fy * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
                pygame.draw.line(self.screen, (255, 215, 0), (center_head_x, center_head_y), (cfx, cfy), 1)

        # Comida
        if game.food:
            fx, fy = game.food
            food_rect = pygame.Rect(
                GAME_AREA_X + fx * GRID_CELL_SIZE + 3,
                GAME_AREA_Y + fy * GRID_CELL_SIZE + 3,
                GRID_CELL_SIZE - 6,
                GRID_CELL_SIZE - 6
            )
            pygame.draw.circle(
                self.screen,
                COLOR_FOOD,
                (food_rect.centerx, food_rect.centery),
                GRID_CELL_SIZE // 2 - 2
            )

        # Cobra
        for idx, (sx, sy) in enumerate(game.snake):
            seg_rect = pygame.Rect(
                GAME_AREA_X + sx * GRID_CELL_SIZE + 2,
                GAME_AREA_Y + sy * GRID_CELL_SIZE + 2,
                GRID_CELL_SIZE - 4,
                GRID_CELL_SIZE - 4
            )
            color = COLOR_SNAKE_HEAD if idx == 0 else COLOR_SNAKE_BODY
            if not game.alive:
                color = COLOR_SNAKE_DEAD
            pygame.draw.rect(self.screen, color, seg_rect, border_radius=6)

        # Informações da partida
        mode_title = "[MODO JOGO / DEMO]" if self.play_mode else "[MODO TREINAMENTO]"
        info_txt = f"{mode_title} Maçãs: {game.score} | Passos: {game.steps_alive}"
        txt_surf = FONT_BODY.render(info_txt, True, COLOR_PLAY_MODE if self.play_mode else COLOR_TEXT_MAIN)
        self.screen.blit(txt_surf, (GAME_AREA_X + 15, GAME_AREA_Y + 15))

        if self.show_vision_rays:
            legend_txt = FONT_TINY.render("Visão Relativa: Vermelho=Frente | Azul=Laterais | Roxo=Diagonais | Amarelo=Comida", True, COLOR_TEXT_MUTED)
            self.screen.blit(legend_txt, (GAME_AREA_X + 15, GAME_AREA_Y + GAME_SIZE - 22))

    def render_neural_panel(self):
        panel_rect = pygame.Rect(NEURAL_AREA_X, NEURAL_AREA_Y, NEURAL_AREA_WIDTH, NEURAL_AREA_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_GRID, panel_rect, width=2, border_radius=10)

        title_text = "Rede Neural em Tempo Real (Demonstração)" if self.play_mode else "Rede Neural em Tempo Real (Cérebro do Agente Líder)"
        title_surf = FONT_TITLE.render(title_text, True, COLOR_TEXT_MAIN)
        self.screen.blit(title_surf, (NEURAL_AREA_X + 20, NEURAL_AREA_Y + 15))

        sub_surf = FONT_SMALL.render("Azul: Sinapses excitatórias (+) | Vermelho: Inibitórias (-) | Amarelo: Neurônios ativos", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (NEURAL_AREA_X + 20, NEURAL_AREA_Y + 40))

        active_agent = self.get_current_active_agent()
        brain = active_agent.brain
        activations = brain.last_activations

        # Camadas: 12 Entradas -> 16 Oculta -> 3 Saídas
        layer_sizes = [brain.input_size, brain.hidden_sizes[0], brain.output_size]
        layer_names = ["12 Sensores Relativos", "Oculta (16 neurônios)", "3 Decisões Relativas"]
        
        margin_x = 180
        margin_y = 80
        avail_w = NEURAL_AREA_WIDTH - 2 * margin_x
        avail_h = NEURAL_AREA_HEIGHT - margin_y - 45

        layer_x = [NEURAL_AREA_X + margin_x + int(i * (avail_w / (len(layer_sizes) - 1))) for i in range(len(layer_sizes))]

        node_coords = []
        for l_idx, count in enumerate(layer_sizes):
            coords = []
            spacing = avail_h / (count + 1)
            for n_idx in range(count):
                ny = NEURAL_AREA_Y + margin_y + int((n_idx + 1) * spacing)
                coords.append((layer_x[l_idx], ny))
            node_coords.append(coords)

        # Conexões/Pesos
        for l_idx in range(len(brain.weights)):
            w_matrix = brain.weights[l_idx]
            from_coords = node_coords[l_idx]
            to_coords = node_coords[l_idx + 1]

            for i, p1 in enumerate(from_coords):
                for j, p2 in enumerate(to_coords):
                    w_val = w_matrix[i, j]
                    if abs(w_val) > 0.30:
                        color = COLOR_WEIGHT_POS if w_val > 0 else COLOR_WEIGHT_NEG
                        width = 2 if abs(w_val) > 1.2 else 1
                        pygame.draw.line(self.screen, color, p1, p2, width)

        # Neurônios e Rótulos
        node_radius = 7
        for l_idx, coords in enumerate(node_coords):
            lbl = FONT_SMALL.render(layer_names[l_idx], True, COLOR_TEXT_MUTED)
            lbl_rect = lbl.get_rect(center=(layer_x[l_idx], NEURAL_AREA_Y + 70))
            self.screen.blit(lbl, lbl_rect)

            layer_acts = activations[l_idx] if l_idx < len(activations) else None

            for n_idx, (nx, ny) in enumerate(coords):
                act_val = layer_acts[n_idx] if layer_acts is not None else 0.0
                
                if act_val > 0.05:
                    intensity = min(255, int(act_val * 255))
                    node_col = (
                        int(COLOR_NODE_INACTIVE[0] + (COLOR_NODE_ACTIVE[0] - COLOR_NODE_INACTIVE[0]) * (intensity / 255)),
                        int(COLOR_NODE_INACTIVE[1] + (COLOR_NODE_ACTIVE[1] - COLOR_NODE_INACTIVE[1]) * (intensity / 255)),
                        int(COLOR_NODE_INACTIVE[2] + (COLOR_NODE_ACTIVE[2] - COLOR_NODE_INACTIVE[2]) * (intensity / 255)),
                    )
                else:
                    node_col = COLOR_NODE_INACTIVE

                r = node_radius + (2 if l_idx == len(layer_sizes) - 1 else 0)
                pygame.draw.circle(self.screen, node_col, (nx, ny), r)
                pygame.draw.circle(self.screen, COLOR_TEXT_MAIN, (nx, ny), r, width=1)

                # Rótulos para neurônios de entrada (à esquerda)
                if l_idx == 0 and n_idx < len(INPUT_LABELS):
                    in_txt = FONT_TINY.render(INPUT_LABELS[n_idx], True, COLOR_TEXT_MUTED)
                    self.screen.blit(in_txt, (nx - 150, ny - 6))

                # Rótulos para neurônios de saída (à direita)
                if l_idx == len(layer_sizes) - 1:
                    action_label = ACTION_NAMES[n_idx]
                    prob = f"{act_val*100:.0f}%"
                    is_max = (act_val == np.max(layer_acts)) if layer_acts is not None else False
                    out_txt = FONT_SMALL.render(f"{action_label} ({prob})", True, COLOR_NODE_ACTIVE if is_max else COLOR_TEXT_MAIN)
                    self.screen.blit(out_txt, (nx + 14, ny - 7))

    def render_controls_and_stats(self):
        ctrl_rect = pygame.Rect(CONTROLS_AREA_X, CONTROLS_AREA_Y, CONTROLS_AREA_WIDTH, CONTROLS_AREA_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, ctrl_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_GRID, ctrl_rect, width=2, border_radius=10)

        alive_count = self.population.get_alive_count()
        gen = self.population.generation
        best_sc = self.population.best_score_ever
        cur_best_sc = max((a.game.score for a in self.population.agents), default=0)

        if self.play_mode:
            stats_txt = (
                f"🎮 MODO JOGO (PLAY) ATIVO   |   Velocidade normal de jogo   |   "
                f"Status: {self.status_message}"
            )
        else:
            stats_txt = (
                f"Geração: {gen}   |   Vivas: {alive_count}/{self.population.size}   |   "
                f"Recorde Histórico: {best_sc} maçãs   |   Melhor da Geração: {cur_best_sc}   |   "
                f"Velocidade: {self.speed_multiplier}x {'(PAUSADO)' if self.paused else ''}"
            )
        stats_surf = FONT_TITLE.render(stats_txt, True, COLOR_TEXT_MAIN)
        self.screen.blit(stats_surf, (CONTROLS_AREA_X + 20, CONTROLS_AREA_Y + 12))

        # Banner de Auto-Verificação
        if not self.play_mode:
            if self.population.recommended_stop:
                alert_box = pygame.Rect(CONTROLS_AREA_X + 20, CONTROLS_AREA_Y + 38, 970, 24)
                pygame.draw.rect(self.screen, (60, 45, 15), alert_box, border_radius=4)
                pygame.draw.rect(self.screen, COLOR_ALERT_BG, alert_box, width=1, border_radius=4)
                alert_txt = FONT_BODY.render(f"★ PONTO IDEAL ALCANÇADO: {self.population.stop_reason} [Clique em 'Salvar Modelo' e vá para 'Modo: JOGAR']", True, (255, 215, 120))
                self.screen.blit(alert_txt, (CONTROLS_AREA_X + 26, CONTROLS_AREA_Y + 42))
            else:
                msg_surf = FONT_BODY.render(f"Status: {self.status_message}", True, COLOR_ACCENT_HOVER)
                self.screen.blit(msg_surf, (CONTROLS_AREA_X + 20, CONTROLS_AREA_Y + 40))

        # Mini gráfico sparkline
        if len(self.population.history_best_scores) >= 2:
            graph_x = CONTROLS_AREA_X + 1020
            graph_y = CONTROLS_AREA_Y + 15
            graph_w = 265
            graph_h = 85
            
            pygame.draw.rect(self.screen, (20, 22, 32), (graph_x, graph_y, graph_w, graph_h), border_radius=6)
            pygame.draw.rect(self.screen, (50, 55, 75), (graph_x, graph_y, graph_w, graph_h), width=1, border_radius=6)

            g_lbl = FONT_TINY.render("Evolução (Recorde por Geração)", True, COLOR_TEXT_MUTED)
            self.screen.blit(g_lbl, (graph_x + 8, graph_y + 4))

            scores = self.population.history_best_scores
            max_val = max(1, max(scores))
            pts = []
            num_pts = len(scores)
            for idx, sc in enumerate(scores):
                px = graph_x + 10 + int(idx * ((graph_w - 20) / (num_pts - 1)))
                py = graph_y + graph_h - 10 - int((sc / max_val) * (graph_h - 30))
                pts.append((px, py))

            if len(pts) >= 2:
                pygame.draw.lines(self.screen, COLOR_ACCENT_HOVER, False, pts, 2)
                for pt in pts[-3:]:
                    pygame.draw.circle(self.screen, COLOR_NODE_ACTIVE, pt, 3)

        # Botões
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            rect = btn["rect"]
            is_hover = rect.collidepoint(mouse_pos)
            btn_id = btn["id"]

            is_active = (
                (btn_id == "mode_toggle" and self.play_mode) or
                (btn_id == "speed_1" and self.speed_multiplier == 1 and not self.play_mode) or
                (btn_id == "speed_5" and self.speed_multiplier == 5 and not self.play_mode) or
                (btn_id == "speed_20" and self.speed_multiplier == 20 and not self.play_mode) or
                (btn_id == "speed_turbo" and self.speed_multiplier == 100 and not self.play_mode)
            )

            if btn_id == "mode_toggle":
                btn_col = COLOR_PLAY_MODE if self.play_mode else (50, 80, 140)
            elif is_active or is_hover:
                btn_col = COLOR_ACCENT
            else:
                btn_col = (42, 45, 62)

            pygame.draw.rect(self.screen, btn_col, rect, border_radius=6)
            pygame.draw.rect(self.screen, (70, 75, 100), rect, width=1, border_radius=6)

            txt = FONT_BODY.render(btn["text"], True, COLOR_TEXT_MAIN)
            txt_rect = txt.get_rect(center=rect.center)
            self.screen.blit(txt, txt_rect)


if __name__ == "__main__":
    app = CobrinhaEducacionalGUI()
    app.run()
