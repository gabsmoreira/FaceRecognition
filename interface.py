import matplotlib
matplotlib.use('TkAgg')

from Tkinter import *
import Tkinter as tk
import time
from datetime import datetime
import training
import face_recognition
import database_creator
from PIL import ImageTk, Image

class Janela_Principal():

    def __init__(self):

        self.window = tk.Tk()
        self.window.geometry("300x300+100+100")
        self.window.title("Face Recoginition")
        self.window.configure(background = 'white')
        self.window.resizable(False, False)

        # Geometria da pagina
        self.window.rowconfigure(0, minsize = 100)
        self.window.rowconfigure(1, minsize = 10)
        self.window.rowconfigure(2, minsize = 10)
        self.window.rowconfigure(3, minsize = 10)
        self.window.rowconfigure(4, minsize = 10)
        self.window.columnconfigure(0, minsize = 10)
        self.window.columnconfigure(1, minsize = 10)




        #Label
        self.Logo = ImageTk.PhotoImage(Image.open("python_logo.jpeg"))
        self.Logo_label = tk.Label(self.window, image = self.Logo, height = 1, width = 1)
        self.Logo_label.grid(row = 0, column = 0, sticky = "nsew")


        #Botoes
        self.button_treinar = tk.Button(self.window, text = "TRAIN", height = 3, width = 30)
        self.button_treinar.grid(row = 1, columnspan = 1)
        self.button_treinar.configure(command = self.treinar)

        self.button_Reconhecimento = tk.Button(self.window, text = "RECOGNIZE", height = 3, width = 30)
        self.button_Reconhecimento.grid(row   = 2, columnspan = 1)
        self.button_Reconhecimento.configure(command = self.reconhecimento)

        self.button_Data_Base = tk.Button(self.window, text = "ADD PERSON", height = 3, width = 30)
        self.button_Data_Base.grid(row   = 3, columnspan = 1)
        self.button_Data_Base.configure(command = self.Base)

        #Label(self.window).grid(row=4, column = 0)
        #e1 = Entry(self.window,)
        #e1.grid(row=4, column=0)


        #self.button_confirm = tk.Button(self.window, text = "CREATE CLASS", height = 4, width = 4)
        #self.button_confirm.grid(row   = 4, column = 1)
        #self.button_confirm.configure(command = self.delete)

    #Loop do codigo
    def iniciar(self):
        self.window.mainloop()

    #Acoes dos botoes
    def treinar(self):
        training.main_training()

    def reconhecimento(self):
        face_recognition.main()

    def Base(self):
        database_creator.main_database()

    def delete(self):
        print("delete")
        #database_creator.delete()

#Loop do codigo
app = Janela_Principal()
app.iniciar()
