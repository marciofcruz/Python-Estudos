import pyautogui
import time
import random
import threading
import sys

# Configurações de segurança: evita cliques fora de controle
pyautogui.FAILSAFE = True  # Mova o mouse para o canto superior esquerdo para parar
pyautogui.PAUSE = 1  # Pausa de 1s entre ações

def mover_mouse():
    """Move o mouse levemente para simular atividade."""
    x, y = pyautogui.position()
    pyautogui.moveTo(x + random.randint(-50, 50), y + random.randint(-20, 20), duration=0.5)

def pressionar_teclas():
    """Pressiona Ctrl ou Shift para atividade no Teams."""
    pyautogui.press('ctrl')  # Mantém "ativo" no Teams sem digitar [web:11]

def alternar_janelas():
    """Alterna entre abas ou janelas (Ctrl+Tab)."""
    pyautogui.hotkey('ctrl', 'tab')

def atividade_continua():
    while True:
        try:
            mover_mouse()
            time.sleep(random.randint(30, 120))  # 30s a 2min entre ações
            pressionar_teclas()
            time.sleep(random.randint(60, 300))  # 1-5min
            if random.random() < 0.3:  # 30% chance de alternar abas
                alternar_janelas()
            print("Atividade simulada. Hora:", time.strftime("%H:%M:%S"))
        except KeyboardInterrupt:
            print("Parando script...")
            sys.exit()

if __name__ == "__main__":
    print("Iniciando simulador. Pressione Ctrl+C para parar.")
    print("Mova o mouse para o canto superior esquerdo para emergência.")
    time.sleep(5)  # 5s para você posicionar o Teams em foco
    atividade_continua()
