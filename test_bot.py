import time
from player import Player
from board_logic_and_mcts.Gomoku_Board import GBoard


def test():
    print("--- ROZPOCZYNAM TEST BOTA ---")

    # 1. Inicjalizacja
    try:
        # Tworzymy bota (Gomoku, plansza 15)
        bot = Player("gomoku", 15)
        print("[OK] Bot zainicjalizowany.")
    except Exception as e:
        print(f"[BLAD] Nie udalo sie stworzyc instancji Player: {e}")
        return

    # 2. Sprawdzenie czy model zaladowany
    if bot.model is not None:
        print("[OK] Model sieci neuronowej zaladowany (nie jest None).")
    else:
        print("[OSTRZEZENIE] Bot dziala w trybie RANDOM (model = None). Sprawdz sciezke do pliku .pth!")

        # ... (wcześniejszy kod)

    # 3. Przygotowanie planszy (NIE PUSTEJ)
    board_size = 15
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]

    # Symulujemy kilka ruchów (żeby bot musiał myśleć)
    board[7][7] = 2  # Przeciwnik na środku
    board[7][8] = 1  # My obok
    board[8][8] = 2  # Przeciwnik

    turn = 3  # 3. tura
    last_opp_move = (8, 8)  # Ostatni ruch przeciwnika (col, row)

    print("\n--- TEST RUCHU (SYTUACJA: SRODEK GRY) ---")
    start = time.time()

    try:
        move = bot.play(board, turn, last_opp_move)
        # ... (reszta kodu bez zmian)
        end = time.time()

        print(f"[SUKCES] Bot zwrocil ruch: {move}")
        print(f"Czas myslenia: {end - start:.2f} sekund")

        # Weryfikacja typu zwracanego
        if isinstance(move, tuple) and len(move) == 2:
            print("[OK] Format ruchu poprawny (row, col).")
        else:
            print(f"[BLAD] Zly format ruchu! Otrzymano: {type(move)}")

    except Exception as e:
        print(f"[BLAD KRYTYCZNY] Bot wyrzucil wyjatek podczas play(): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()