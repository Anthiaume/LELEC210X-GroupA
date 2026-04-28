import tkinter as tk
import time


def mettre_a_jour():
    heure_actuelle = time.ctime()
    label.config(text=heure_actuelle)
    fenetre.after(1000, mettre_a_jour)


fenetre = tk.Tk()
fenetre.title("Horloge")
fenetre.resizable(False, False)

# Set x and y maximal size of the screen
screen_width = fenetre.winfo_screenwidth()
screen_height = fenetre.winfo_screenheight()
label = tk.Label(
    fenetre,
    font=("Courier", 70, "bold"),
    fg="#00ff88",
    bg="#1a1a2e",
    # padx = screen_width // 2 - textwidth // 2,
    padx=20,
    pady=20,
)
label.pack()

mettre_a_jour()
fenetre.mainloop()
