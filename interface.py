import Tkinter as tk
import time
from datetime import datetime
import training
import face_recognition
import database_creator

class Janela_Principal():

    def __init__(self):

        self.window = tk.Tk()
        self.window.geometry("250x450+100+100")
        self.window.title("Face Recoginition")
        self.window.configure(background = 'white')
        self.window.resizable(False, False)

        # Geometria da pagina
        self.window.rowconfigure(0, minsize = 100)
        self.window.rowconfigure(1, minsize = 100)
        self.window.rowconfigure(2, minsize = 100)
        self.window.rowconfigure(3, minsize = 100)
        self.window.columnconfigure(0, minsize = 250)

        #Label
#        self.Logo = tk.PhotoImage(file = "python_logo.png")
#        self.Logo_label = tk.Label(self.window, image = self.Logo, height = 1, width = 1)
#        self.Logo_label.grid(row = 0, column = 0, sticky = "nsew")

        #Botoes
        self.button_treinar = tk.Button(self.window, text = "TRAIN", height = 3, width = 30)
        self.button_treinar.grid(row = 1, column = 0)
        self.button_treinar.configure(command = self.treinar)

        self.button_Reconhecimento = tk.Button(self.window, text = "RECOGNIZE", height = 3, width = 30)
        self.button_Reconhecimento.grid(row   = 2, column = 0)
        self.button_Reconhecimento.configure(command = self.reconhecimento)


        self.button_Data_Base = tk.Button(self.window, text = "ADD PERSON", height = 3, width = 30)
        self.button_Data_Base.grid(row   = 3, column = 0)
        self.button_Data_Base.configure(command = self.Base)


    def iniciar(self):
        self.window.mainloop()

    def treinar(self):
        training.main_training()

    def reconhecimento(self):
        face_recognition.main()

    def Base(self):
        database_creator.main_database()

app = Janela_Principal()
app.iniciar()
