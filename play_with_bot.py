import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

# Importy z Twojego projektu
from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.Node import Node
from board_logic_and_mcts.MCTS_Neural import MCTS_Neural
from CNN.nn_model import PolicyValueNet

# --- KONFIGURACJA ---
BOARD_SIZE = 15
MODEL_PATH = "CNN/final_gomoku_model.pth"  # Upewnij się, że masz ten plik!
HUMAN_STARTS = False  # True = Ty zaczynasz (Czarne), False = Bot zaczyna


class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gomoku: Human vs AlphaZero")

        self.cell_size = 30
        self.margin = 20
        self.board_size = BOARD_SIZE

        # Logika gry
        self.game = GBoard(self.board_size)
        self.game_over = False
        self.human_turn = HUMAN_STARTS

        # Ładowanie bota
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet(board_size=self.board_size, channels=128).to(self.device)

        try:
            state = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print("Model załadowany pomyślnie!")
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            self.model = None

        # Canvas
        canvas_size = 2 * self.margin + (self.board_size - 1) * self.cell_size
        self.canvas = tk.Canvas(master, width=canvas_size, height=canvas_size, bg="#eebb99")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()

        if not self.human_turn:
            self.master.after(500, self.bot_move)

    def draw_board(self):
        self.canvas.delete("all")
        # Siatka
        for i in range(self.board_size):
            start = self.margin
            end = self.margin + (self.board_size - 1) * self.cell_size
            p = self.margin + i * self.cell_size
            self.canvas.create_line(start, p, end, p)
            self.canvas.create_line(p, start, p, end)

        # Kamienie
        for y in range(self.board_size):
            for x in range(self.board_size):
                val = self.game.board[y][x]
                if val != 0:
                    cx = self.margin + x * self.cell_size
                    cy = self.margin + y * self.cell_size
                    color = "black" if val == 1 else "white"
                    self.canvas.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill=color)

                    # Oznaczenie ostatniego ruchu
                    if self.game.last_move == (x, y):
                        self.canvas.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, fill="red")

    def on_click(self, event):
        if self.game_over or not self.human_turn:
            return

        # Przeliczanie kliknięcia na współrzędne
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        if 0 <= col < self.board_size and 0 <= row < self.board_size:
            if self.game.isLegalMove(col, row):
                self.make_move(col, row, 1 if HUMAN_STARTS else 2)  # Ty grasz 1 lub 2
                self.human_turn = False
                self.master.after(100, self.bot_move)

    def bot_move(self):
        if self.game_over: return

        print("Bot myśli...")
        bot_player = 2 if HUMAN_STARTS else 1

        # Używamy Twojego MCTS_Neural
        root = Node(deepcopy(self.game), None)
        mcts = MCTS_Neural(root, bot_player, self.model, self.device, simulation_limit=400, timeout=4.0)
        best_node = mcts.best_move()

        if best_node and best_node.board.last_move:
            col, row = best_node.board.last_move
            self.make_move(col, row, bot_player)
            self.human_turn = True
        else:
            print("Bot nie ma ruchu!")

    def make_move(self, col, row, player):
        self.game.makeMove(col, row, player)
        self.draw_board()

        if self.game.isWon(col, row, player):
            winner = "Człowiek" if self.human_turn else "Bot"
            messagebox.showinfo("Koniec gry", f"Wygrał: {winner}!")
            self.game_over = True
        elif self.game.isBoardFull():
            messagebox.showinfo("Koniec gry", "Remis!")
            self.game_over = True


if __name__ == "__main__":
    root = tk.Tk()
    gui = GomokuGUI(root)
    root.mainloop()