'''Example of how to use the grid() method to create a GUI layout'''
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image,ImageTk

class MyMainWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        lengh_size_path = 50

        self.title("")
        self.maxsize(1240,  800)  # width x height

        # Create left frame and Notebook
        left_frame  =  ttk.Frame(self,  width=600,  height=  800)
        # left_frame["padding"] = 100
        # left_frame["borderwidth"] = 100
        # left_frame["relief"] = 'sunken'
        left_frame.grid(row=0,  column=0,  padx=10,  pady=5)
        left_Notebook = ttk.Notebook(left_frame)

        # Create left Notebook and tabs inside
        tab_conf = ttk.Frame(left_Notebook, padding = 10)
        left_Notebook.add(tab_conf, text='Configuration')
        tab_test = ttk.Frame(left_Notebook, padding = 10)
        left_Notebook.add(tab_test, text='Tests')
        tab_batch = ttk.Frame(left_Notebook, padding = 10)
        left_Notebook.add(tab_batch, text='batch')
        left_Notebook.pack(expand=1, fill="both", side = 'top')

        # Create buttons for defining path in configuration tab
        label_rep_1 = ttk.Label(tab_conf, text="Holo file path :")
        self.holo_file_path = ttk.Entry(tab_conf, width = lengh_size_path)
        button_browse_holo_file = ttk.Button(tab_conf, text="browse...", command=lambda: self.browseFile(self.holo_file_path))

        label_rep_2 = ttk.Label(tab_conf, text="Holograms directory :")
        self.holo_directory_path = ttk.Entry(tab_conf, width = lengh_size_path)
        button_path_2 = ttk.Button(tab_conf, text="browse...", command=lambda: self.browseDirectory(self.holo_directory_path))

        label_rep_3 = ttk.Label(tab_conf, text="Result file path :")
        self.result_file_path = ttk.Entry(tab_conf, width = lengh_size_path)
        button_path_3 = ttk.Button(tab_conf, text="browse...", command=lambda: self.browseNewFile(self.result_file_path))

        label_lambda = ttk.Label(tab_conf, text="Wavelengh:")
        self.wavelengh = ttk.Entry(tab_conf)
        label_index = ttk.Label(tab_conf, text="Optical index:")
        self.opt_index = ttk.Entry(tab_conf)
        label_image_width = ttk.Label(tab_conf, text="Image width:")
        self.image_width = ttk.Entry(tab_conf)
        label_image_height = ttk.Label(tab_conf, text="Image height:")
        self.image_height = ttk.Entry(tab_conf)
        label_pixel_size = ttk.Label(tab_conf, text="Pixel size:")
        self.pix_size = ttk.Entry(tab_conf)
        label_magnification = ttk.Label(tab_conf, text="Magnification :")
        self.magnification = ttk.Entry(tab_conf)
        label_start_propag_distance = ttk.Label(tab_conf, text="Start propagation distance:")
        self.start_propag_distance = ttk.Entry(tab_conf)
        label_propag_step = ttk.Label(tab_conf, text="Propagation step:")
        self.propag_step = ttk.Entry(tab_conf)
        label_number_of_plane = ttk.Label(tab_conf, text="number of propag plane:")
        self.numer_of_plane = ttk.Entry(tab_conf)

        label_focus_type = ttk.Label(tab_conf, text="Type of focus:")
        self.focus = tk.StringVar()
        combobox_focus = ttk.Combobox(tab_conf, textvariable=self.focus)
        combobox_focus['values'] = ('SUM_OF_LAPLACIAN', 'SUM_OF_VARIANCE', 'TENENGRAD', 'SUM_OF_INTENSITY')
        combobox_focus.bind('<<ComboboxSelected>>', lambda e: print(self.focus.get))

        #Grid positions in configuration tab
        label_rep_1.grid(row=0, column=0, sticky="e")
        self.holo_file_path.grid(row=0, column=1)
        button_browse_holo_file.grid(row=0, column=2)

        label_rep_2.grid(row=1, column=0, sticky="e")
        self.holo_directory_path.grid(row=1, column=1)
        button_path_2.grid(row=1, column=2)

        label_rep_3.grid(row=2, column=0, sticky="e")
        self.result_file_path.grid(row=2, column=1)
        button_path_3.grid(row=2, column=2)
        
        label_lambda.grid(row=3, column=0, sticky="e")
        self.wavelengh.grid(row=3, column=1, sticky="w")
        label_index.grid(row=4, column=0, sticky="e")
        self.opt_index.grid(row=4, column=1, sticky="w")
        label_image_width.grid(row=5, column=0, sticky="e")
        self.image_width.grid(row=5, column=1, sticky="w")
        label_image_height.grid(row=6, column=0, sticky="e")
        self.image_height.grid(row=6, column=1, sticky="w")
        label_pixel_size.grid(row=7, column=0, sticky="e")
        self.pix_size.grid(row=7, column=1, sticky="w")
        label_magnification.grid(row=8, column=0, sticky="e")
        self.magnification.grid(row=8, column=1, sticky="w")
        label_start_propag_distance.grid(row=9, column=0, sticky="e")
        self.start_propag_distance.grid(row=9, column=1, sticky="w")
        label_propag_step.grid(row=10, column=0, sticky="e")
        self.propag_step.grid(row=10, column=1, sticky="w")
        label_number_of_plane.grid(row=11, column=0, sticky="e")
        self.numer_of_plane.grid(row=11, column=1, sticky="w")
        label_focus_type.grid(row=12, column=0, sticky="e")
        combobox_focus.grid(row=12, column=1, sticky="w")


		# Create buttons in test tab
        btn_open_Holo = ttk.Button(tab_test, text=" open holo ")
        btn_open_Holo.grid(row=0, column=0, sticky="w")
        btn_close_Holo = ttk.Button(tab_test, text=" close holo ")
        btn_close_Holo.grid(row=0, column=1, sticky="w")
        btn_suppress_mean = ttk.Button(tab_test, text=" suppress mean holo ")
        btn_suppress_mean.grid(row=1, column=0, sticky="w")
        btn_volume_propagation = ttk.Button(tab_test, text=" propagate volume ")
        btn_volume_propagation.grid(row=2, column=0, sticky="w")

		# Create right frame and Notebook
        right_frame  =  ttk.Frame(self,  width=600,  height=800)
        right_frame.grid(row=0,  column=1,  padx=10,  pady=5)
        right_tabControl = ttk.Notebook(right_frame)
        
		#Create tab in right frame
        right_tab_holo = ttk.Frame(right_tabControl)
        right_tabControl.add(right_tab_holo, text='Holo')
        right_tab_data = ttk.Frame(right_frame)
        right_tabControl.add(right_tab_data, text='Data')
        right_tabControl.pack(expand=1, fill="both", side = 'top')
    
    # Fonction appelée lorsque le bouton "Parcourir..." est cliqué
    def browseDirectory(self, edit):
        directory = filedialog.askdirectory()
        if directory:
            edit.delete(0, tk.END)
            edit.insert(0, directory)

    # Fonction appelée lorsque le bouton "Parcourir..." est cliqué
    def browseFile(self, edit):
        name = filedialog.askopenfilename()
        if name:
            edit.delete(0, tk.END)
            edit.insert(0, name)

    # Fonction appelée lorsque le bouton "Parcourir..." est cliqué
    def browseNewFile(self, edit):
        name = filedialog.asksaveasfilename()
        if name:
            edit.delete(0, tk.END)
            edit.insert(0, name)
    
	# #fonction de mise à jour de la comboBox focus
    # def update_foc(self):
    



if __name__ == "__main__":
    window = MyMainWindow()
    window.mainloop()