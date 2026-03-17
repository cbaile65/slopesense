import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
import Defect_Masking


class FlowcheckGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FLOWCHECK")
        self.root.configure(bg="white")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.escape_fullscreen)

        self.current_w = 640
        self.current_h = 480

        self.last_cw = 0
        self.last_ch = 0

        # ==========================================
        # 1. MAIN WINDOW GRID LAYOUT
        # ==========================================
        # Initial grid config; dynamic sizing is handled by apply_sku_layout()
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        # ==========================================
        # 2. TOP ROW: Header Frame (Left) & Info Frame (Right)
        # ==========================================
        # -- LEFT SIDE: Logo & Version --
        self.header_frame = tk.Frame(self.root, bg="white")
        self.header_frame.grid(row=0, column=0, sticky="nw", padx=40, pady=(20, 10))

        logo_path = os.path.join("files", "flowcheck_logo.png")
        try:
            logo_img = Image.open(logo_path)
            target_height = 100
            aspect_ratio = logo_img.width / logo_img.height
            target_width = int(target_height * aspect_ratio)

            logo_img = logo_img.resize((target_width, target_height), Image.LANCZOS)
            self.logo_imgtk = ImageTk.PhotoImage(logo_img)
            self.title_label = tk.Label(self.header_frame, image=self.logo_imgtk, bg="white")
        except Exception as e:
            print(f"Error loading logo: {e}. Falling back to text.")
            self.title_label = tk.Label(self.header_frame, text="FLOWCHECK", font=("Arial", 36), bg="white")

        self.title_label.pack(side="left")

        self.version_label = tk.Label(
            self.header_frame,
            text="V0.1",
            font=("Arial", 24, "bold"),
            bg="white",
            fg="black"
        )
        self.version_label.pack(side="left", padx=(15, 0), anchor="s", pady=(0, 15))

        # -- RIGHT SIDE: Info Text --
        self.info_frame = tk.Frame(self.root, bg="white")

        self.lbl_cam_height = tk.Label(self.info_frame, text="Camera Height: --", bg="white", fg="black")
        self.lbl_tub_detected = tk.Label(self.info_frame, text="Tub Detected: --", bg="white", fg="black")
        self.lbl_defects = tk.Label(self.info_frame, text="Defects Detected: --", bg="white", fg="black")
        self.lbl_units = tk.Label(self.info_frame, text="Units Scanned: --", bg="white", fg="black")

        self.lbl_cam_height.pack(anchor="w", pady=6)
        self.lbl_tub_detected.pack(anchor="w", pady=6)
        self.lbl_defects.pack(anchor="w", pady=6)
        self.lbl_units.pack(anchor="w", pady=6)

        # ==========================================
        # 3. MIDDLE ROW: Video Feed
        # ==========================================
        self.video_container = tk.Frame(self.root, bg="white")
        self.video_label = tk.Label(self.video_container, bg="white")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.video_label.bind("<Button-1>", self.on_video_click)
        self.root.bind("<r>", self.on_r_keypress)

        # ==========================================
        # 4. BOTTOM ROW: Main Screen Buttons
        # ==========================================
        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=40, pady=(20, 40))

        for col in range(6):
            self.button_frame.columnconfigure(col, weight=1, uniform="btn_cols")

        button_config = [
            {"text": "Select SKU", "command": self.open_sku_menu, "bg": "#A9A9A9"},
            {"text": "", "command": None, "bg": "#A9A9A9"},
            {"text": "", "command": None, "bg": "#A9A9A9"},
            {"text": "", "command": None, "bg": "#A9A9A9"},
            {"text": "Toggle Debug", "command": None, "bg": "#A9A9A9"},
            {"text": "STOP", "command": self.on_closing, "bg": "red"}
        ]

        for i, config in enumerate(button_config):
            btn = tk.Button(
                self.button_frame,
                text=config["text"],
                font=("Arial", 16, "bold"),
                bg=config["bg"],
                fg="black",
                height=3,
                command=config["command"]
            )
            px = (0, 15) if i < 5 else (0, 0)
            btn.grid(row=0, column=i, sticky="nsew", padx=px)

        # ==========================================
        # 5. Build Submenus & Set Default Layout
        # ==========================================
        self.create_sku_menu()

        # Set "Strada (Shower Base)" as the default layout on boot
        self.apply_sku_layout("Strada\n(Shower Base)")
        self.root.update_idletasks()

        # ==========================================
        # 6. Initialize Camera & Threading
        # ==========================================
        self.camera = Defect_Masking.DefectDetector()

        self.latest_frame = None
        self.last_drawn_frame = None
        self.is_running = True

        self.capture_thread = threading.Thread(target=self.frame_capture_thread, daemon=True)
        self.capture_thread.start()

        self.update_video()

    def frame_capture_thread(self):
        """Runs continuously in the background to fetch frames."""
        while self.is_running:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    self.latest_frame = frame
            except Exception as e:
                print(f"Camera thread error: {e}")
            time.sleep(0.01)

    # ==========================================
    # SUBMENU LOGIC
    # ==========================================
    def create_sku_menu(self):
        self.sku_frame = tk.Frame(self.root, bg="white")

        for col in range(3):
            self.sku_frame.columnconfigure(col, weight=1, uniform="sku_grid_cols")
        for row in range(2):
            self.sku_frame.rowconfigure(row, weight=1, uniform="sku_grid_rows")

        sku_button_config = [
            {"text": "Strada\n(Shower Base)", "command": lambda: self.select_sku("Strada\n(Shower Base)"),
             "bg": "#E0E0E0"},
            {"text": "(Skirted Tub)", "command": lambda: self.select_sku("(Skirted Tub)"), "bg": "#E0E0E0"},
            {"text": "(Tub-Shower)", "command": lambda: self.select_sku("(Tub-Shower)"), "bg": "#E0E0E0"},
            {"text": "", "command": None, "bg": "#E0E0E0"},
            {"text": "*Raw Scan*", "command": None, "bg": "#E0E0E0"},
            {"text": "Back", "command": self.close_sku_menu, "bg": "#A9A9A9"}
        ]

        for i, config in enumerate(sku_button_config):
            r = i // 3
            c = i % 3

            btn = tk.Button(
                self.sku_frame,
                text=config["text"],
                font=("Arial", 24, "bold"),
                bg=config["bg"],
                fg="black",
                command=config["command"]
            )
            btn.grid(row=r, column=c, sticky="nsew", padx=15, pady=15)

    def select_sku(self, sku_name):
        self.apply_sku_layout(sku_name)
        self.close_sku_menu()

    def apply_sku_layout(self, sku_name):
        """Dynamically adjusts the grid proportions and text based on the selected SKU."""
        if sku_name in ["Strada\n(Shower Base)", "(Skirted Tub)"]:
            # EXPANDED LAYOUT: 60% / 40% split to prevent text cutoff
            self.root.columnconfigure(0, weight=3, uniform="expanded_cols")
            self.root.columnconfigure(1, weight=2, uniform="expanded_cols")

            # Push camera container as far left as possible
            self.video_container.grid(row=1, column=0, sticky="nsew", padx=(10, 10), pady=10)

            # Center text in row 1, slightly adjust font size
            large_font = ("Arial", 28, "bold")
            self.lbl_cam_height.config(font=large_font)
            self.lbl_tub_detected.config(font=large_font)
            self.lbl_defects.config(font=large_font)
            self.lbl_units.config(font=large_font)

            self.info_frame.grid_configure(row=1, column=1, rowspan=1, sticky="w", padx=(10, 20), pady=0)

        elif sku_name == "(Tub-Shower)":
            # DEFAULT LAYOUT: 50/50 screen split
            self.root.columnconfigure(0, weight=1, uniform="main_cols")
            self.root.columnconfigure(1, weight=1, uniform="main_cols")

            # Restore standard camera container padding
            self.video_container.grid(row=1, column=0, sticky="nsew", padx=(40, 20), pady=10)

            # Move text back to top right
            default_font = ("Arial", 24, "bold")
            self.lbl_cam_height.config(font=default_font)
            self.lbl_tub_detected.config(font=default_font)
            self.lbl_defects.config(font=default_font)
            self.lbl_units.config(font=default_font)

            self.info_frame.grid_configure(row=0, column=1, rowspan=2, sticky="nw", padx=(20, 40), pady=(30, 10))

    def open_sku_menu(self):
        self.video_container.grid_remove()
        self.button_frame.grid_remove()
        self.info_frame.grid_remove()

        self.sku_frame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew", padx=25, pady=(0, 25))

    def close_sku_menu(self):
        self.sku_frame.grid_remove()

        self.video_container.grid()
        self.button_frame.grid()
        self.info_frame.grid()

    # ==========================================
    # CORE FUNCTIONS
    # ==========================================
    def escape_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

    def on_video_click(self, event):
        scale_x = 640 / self.current_w
        scale_y = 480 / self.current_h
        native_x = int(event.x * scale_x)
        native_y = int(event.y * scale_y)
        self.camera.register_click(native_x, native_y)

    def on_r_keypress(self, event):
        self.camera.reset_depth()

    def update_video(self):
        # 1. Update UI Labels from Camera Data
        if self.camera.ref_depth is not None:
            self.lbl_cam_height.config(text=f"Camera Height: {self.camera.ref_depth:.3f} m")
        else:
            self.lbl_cam_height.config(text="Camera Height: --")

        # 2. Update Video Feed
        frame = self.latest_frame

        if frame is not None and frame is not self.last_drawn_frame:
            self.last_drawn_frame = frame

            cw = self.video_container.winfo_width()
            ch = self.video_container.winfo_height()

            if cw > 10 and ch > 10:
                if cw != self.last_cw or ch != self.last_ch:
                    aspect = 640 / 480
                    container_aspect = cw / ch

                    if container_aspect > aspect:
                        self.current_h = ch
                        self.current_w = int(ch * aspect)
                    else:
                        self.current_w = cw
                        self.current_h = int(cw / aspect)

                    self.last_cw = cw
                    self.last_ch = ch

                frame_resized = cv2.resize(frame, (self.current_w, self.current_h), interpolation=cv2.INTER_LINEAR)
                cv2_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv2_img)
                imgtk = ImageTk.PhotoImage(image=pil_img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def on_closing(self):
        self.is_running = False
        self.camera.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowcheckGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()