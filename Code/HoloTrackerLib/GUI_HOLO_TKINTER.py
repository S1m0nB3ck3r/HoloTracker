import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread, Event
import os
import json
import numpy as np
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

stop_event = Event()
allocated = False

class HoloApp:
    def __init__(self, root):
        self.root = root
        root.title("Hologram Analysis GUI")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.params = {}
        self.create_widgets()
        self.load_parameters()

    def create_widgets(self):
        # Left Panel for Inputs
        left_panel = tk.Frame(self.root)
        left_panel.grid(row=0, column=0, padx=10, pady=10)

        self.fields = {
            "holograms_directory_path": ("Holograms Directory", tk.Entry),
            "result_filename": ("Result Filename", tk.Entry),
            "medium_wavelenght": ("Medium Wavelength (m)", tk.Entry),
            "cam_magnification": ("Camera Magnification", tk.Entry),
            "cam_nb_pix_X": ("Cam nb_pix_X", tk.Entry),
            "cam_nb_pix_Y": ("Cam nb_pix_Y", tk.Entry),
            "nb_plane": ("Nb Plane", tk.Entry),
            "cam_pix_size": ("Pixel Size (m)", tk.Entry),
            "frequency_min": ("Frequency Min", tk.Entry),
            "frequency_max": ("Frequency Max", tk.Entry),
            "focus_smooth_size": ("Focus Smooth Size", tk.Entry),
            "nb_StdVar_threshold": ("Nb StdVar Threshold", tk.Entry),
            "type_Threshold": ("Threshold Type", ttk.Combobox),
            "n_connectivity": ("Connectivity (6/18/26)", tk.Entry),
            "particle_filter_size_min": ("Particle Size Min", tk.Entry),
            "particle_filter_size_max": ("Particle Size Max", tk.Entry),
        }

        self.input_widgets = []
        row = 0
        for key, (label, widget_type) in self.fields.items():
            tk.Label(left_panel, text=label).grid(row=row, column=0, sticky="e")
            if key == "type_Threshold":
                entry = widget_type(left_panel, values=["value", "nb_std_var"])
            else:
                entry = widget_type(left_panel)
            entry.grid(row=row, column=1)
            self.params[key] = entry
            self.input_widgets.append(entry)
            if "path" in key or "filename" in key:
                browse_button = tk.Button(left_panel, text="Browse", command=lambda k=key: self.browse_file_or_dir(k))
                browse_button.grid(row=row, column=2)
                self.input_widgets.append(browse_button)
            row += 1

        self.show_images_var = tk.BooleanVar(value=False)
        self.show_images_cb = tk.Checkbutton(left_panel, text="Afficher les images (batch)", variable=self.show_images_var)
        self.show_images_cb.grid(row=row, column=0, columnspan=2, sticky="w")
        self.input_widgets.append(self.show_images_cb)
        row += 1

        tk.Label(left_panel, text="Test Hologram").grid(row=row, column=0, sticky="e")
        self.test_holo_cb = ttk.Combobox(left_panel, values=[])
        self.test_holo_cb.grid(row=row, column=1)
        self.params['test_hologram'] = self.test_holo_cb
        self.test_holo_cb.bind("<Button-1>", self.update_hologram_list)
        self.input_widgets.append(self.test_holo_cb)
        row += 1

        self.test_button = tk.Button(left_panel, text="Start Test Mode", command=self.start_test_mode)
        self.test_button.grid(row=row, column=0)
        self.quit_test_button = tk.Button(left_panel, text="Quit Test Mode", command=self.quit_test_mode)
        self.quit_test_button.grid(row=row, column=1)
        row += 1
        self.run_batch_button = tk.Button(left_panel, text="Start Batch", command=self.start_batch_mode)
        self.run_batch_button.grid(row=row, column=0)
        self.stop_button = tk.Button(left_panel, text="Stop Batch", command=self.stop_batch)
        self.stop_button.grid(row=row, column=1)
        row += 1

        self.status_label = tk.Label(left_panel, text="Ready", anchor="w", fg="blue")
        self.status_label.grid(row=row, column=0, columnspan=3, sticky="ew")

        self.status_text = tk.Text(left_panel, height=10, width=50)
        self.status_text.grid(row=row+1, column=0, columnspan=3, sticky="ew")

        # Right Panel for Image Display
        right_panel = tk.Frame(self.root)
        right_panel.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(right_panel, text="Processed Image").pack()
        self.canvas = tk.Canvas(right_panel, width=400, height=400)
        self.canvas.pack()

        # Display Mode Selection
        tk.Label(right_panel, text="Display Mode").pack()
        self.display_mode = ttk.Combobox(right_panel, values=[
            "hologram", "plane number", "XY max plane", "XY sum plane",
            "XZ max plane", "XZ sum plane", "YZ max plane", "YZ sum plane"
        ])
        self.display_mode.pack()
        self.display_mode.bind("<<ComboboxSelected>>", self.on_display_mode_change)

        # Plane Number Selection
        tk.Label(right_panel, text="Plane Number").pack()
        self.plane_number = tk.Spinbox(right_panel, from_=0, to=9, state="readonly")
        self.plane_number.pack()

    def browse_file_or_dir(self, key):
        if "directory" in key:
            path = filedialog.askdirectory()
        else:
            path = filedialog.asksaveasfilename()
        if path:
            self.params[key].delete(0, tk.END)
            self.params[key].insert(0, path)
            if key == "holograms_directory_path":
                self.update_hologram_list()

    def update_params(self):
        out = {}
        for key, widget in self.params.items():
            if isinstance(widget, ttk.Combobox):
                out[key] = widget.get()
            else:
                try:
                    val = widget.get()
                    if val.replace('.', '', 1).isdigit():
                        out[key] = float(val) if '.' in val else int(val)
                    else:
                        out[key] = val
                except:
                    out[key] = val
        out["show_images"] = self.show_images_var.get()
        return out

    def update_hologram_list(self, event=None):
        path = self.params["holograms_directory_path"].get()
        if os.path.isdir(path):
            bmps = [f for f in os.listdir(path) if f.lower().endswith('.bmp')]
            self.test_holo_cb["values"] = bmps
            if bmps:
                self.test_holo_cb.set(bmps[0])
            nb_plane = int(self.params["nb_plane"].get())
            self.plane_number.config(to=nb_plane-1)

    def start_test_mode(self):
        global allocated
        self.update_hologram_list()
        params = self.update_params()
        if not allocated:
            self.set_status("Allocating memory...")
            # allocate_memory(params)
            allocated = True
        self.set_status(f"Processing test hologram: {params['test_hologram']}")
        self.disable_test_mode_buttons()
        # result = process_test_hologram(params)
        # self.display_image(result)

    def quit_test_mode(self):
        global allocated
        if allocated:
            self.set_status("Deallocating memory...")
            # deallocate_memory()
            allocated = False
            self.set_status("Test mode exited.")
        self.enable_all_buttons()

    def start_batch_mode(self):
        params = self.update_params()
        stop_event.clear()
        self.disable_batch_mode_buttons()
        self.set_status("Starting batch...")
        Thread(target=self.run_batch, args=(params,)).start()

    def run_batch(self, params):
        # process_batch_holograms(params, stop_event, self.display_image)
        self.set_status("Batch done or interrupted.")
        self.enable_all_buttons()

    def stop_batch(self):
        stop_event.set()
        self.set_status("Batch stop requested.")

    def disable_inputs(self):
        for widget in self.input_widgets:
            widget.configure(state='disabled')

    def disable_test_mode_buttons(self):
        for widget in self.input_widgets:
            widget.configure(state='disabled')
        self.test_button.configure(state='disabled')
        self.run_batch_button.configure(state='disabled')
        self.stop_button.configure(state='disabled')
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

    def disable_batch_mode_buttons(self):
        for widget in self.input_widgets:
            widget.configure(state='disabled')
        self.test_button.configure(state='disabled')
        self.quit_test_button.configure(state='disabled')
        self.run_batch_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

    def enable_all_buttons(self):
        for widget in self.input_widgets:
            widget.configure(state='normal')
        self.test_button.configure(state='normal')
        self.quit_test_button.configure(state='normal')
        self.run_batch_button.configure(state='normal')
        self.stop_button.configure(state='normal')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def display_image(self, np_img):
        img = Image.fromarray(np.uint8(255 * (np_img - np_img.min()) / (np_img.ptp() + 1e-8)))
        img = img.resize((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def set_status(self, message):
        self.status_label.config(text=message)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        print(message)

    def on_display_mode_change(self, event):
        mode = self.display_mode.get()
        self.set_status(f"Display mode changed to: {mode}")
        # Handle display mode change logic here

    def save_parameters(self):
        params = self.update_params()
        with open('parameters.json', 'w') as f:
            json.dump(params, f)

    def load_parameters(self):
        try:
            with open('parameters.json', 'r') as f:
                params = json.load(f)
                for key, value in params.items():
                    if key in self.params:
                        self.params[key].delete(0, tk.END)
                        self.params[key].insert(0, value)
                if "show_images" in params:
                    self.show_images_var.set(params["show_images"])
        except FileNotFoundError:
            pass

    def on_closing(self):
        self.save_parameters()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = HoloApp(root)
    root.mainloop()
