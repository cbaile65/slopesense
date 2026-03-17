import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
import Defect_Masking
import HardwareManager


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
            {"text": "Manual Motor\nControl", "command": self.open_motor_menu, "bg": "#A9A9A9"},
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
        self.hw = HardwareManager.HardwareManager()

        self.create_sku_menu()
        self.create_motor_menu()

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
        while self.is_running:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    self.latest_frame = frame
            except Exception as e:
                print(f"Camera thread error: {e}")
            time.sleep(0.01)

    # ==========================================
    # SKU SUBMENU LOGIC
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
        if sku_name in ["Strada\n(Shower Base)", "(Skirted Tub)"]:
            self.root.columnconfigure(0, weight=3, uniform="expanded_cols")
            self.root.columnconfigure(1, weight=2, uniform="expanded_cols")

            self.video_container.grid(row=1, column=0, sticky="nsew", padx=(10, 10), pady=10)

            large_font = ("Arial", 28, "bold")
            self.lbl_cam_height.config(font=large_font)
            self.lbl_tub_detected.config(font=large_font)
            self.lbl_defects.config(font=large_font)
            self.lbl_units.config(font=large_font)

            self.info_frame.grid_configure(row=1, column=1, rowspan=1, sticky="w", padx=(10, 20), pady=0)

        elif sku_name == "(Tub-Shower)":
            self.root.columnconfigure(0, weight=1, uniform="main_cols")
            self.root.columnconfigure(1, weight=1, uniform="main_cols")

            self.video_container.grid(row=1, column=0, sticky="nsew", padx=(40, 20), pady=10)

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
    # MOTOR SUBMENU LOGIC
    # ==========================================
    def create_motor_menu(self):
        self.motor_frame = tk.Frame(self.root, bg="white")

        self.motor_frame.columnconfigure(0, weight=1)
        self.motor_frame.columnconfigure(1, weight=3)
        self.motor_frame.columnconfigure(2, weight=1)

        self.motor_frame.rowconfigure(0, weight=1)
        self.motor_frame.rowconfigure(1, weight=3)
        self.motor_frame.rowconfigure(2, weight=1)

        # -- Top Row: Rotations --
        # CW on Top Left
        btn_cw = tk.Button(self.motor_frame, text="⟳", font=("Arial", 42, "bold"), bg="#E0E0E0", padx=15, pady=5)
        btn_cw.grid(row=0, column=2, sticky="ne", padx=40, pady=30)
        btn_cw.bind("<ButtonPress-1>", lambda e: self.hw.rotate_cw(True))
        btn_cw.bind("<ButtonRelease-1>", lambda e: self.hw.rotate_cw(False))

        # CCW on Top Right
        btn_ccw = tk.Button(self.motor_frame, text="⟲", font=("Arial", 42, "bold"), bg="#E0E0E0", padx=15, pady=5)
        btn_ccw.grid(row=0, column=0, sticky="nw", padx=40, pady=30)
        btn_ccw.bind("<ButtonPress-1>", lambda e: self.hw.rotate_ccw(True))
        btn_ccw.bind("<ButtonRelease-1>", lambda e: self.hw.rotate_ccw(False))

        # -- Center Row: D-Pad --
        dpad_container = tk.Frame(self.motor_frame, bg="white")
        dpad_container.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        dpad_container.columnconfigure(0, weight=5)
        dpad_container.columnconfigure(1, weight=4)
        dpad_container.columnconfigure(2, weight=5)

        for i in range(3):
            dpad_container.rowconfigure(i, weight=1, uniform="dpad_rows")

        dpad_font = ("Arial", 16, "bold")

        btn_up = tk.Button(dpad_container, text="UP", font=dpad_font, bg="#E0E0E0")
        btn_up.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        btn_up.bind("<ButtonPress-1>", lambda e: self.hw.move_up(True))
        btn_up.bind("<ButtonRelease-1>", lambda e: self.hw.move_up(False))

        btn_down = tk.Button(dpad_container, text="DOWN", font=dpad_font, bg="#E0E0E0")
        btn_down.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        btn_down.bind("<ButtonPress-1>", lambda e: self.hw.move_down(True))
        btn_down.bind("<ButtonRelease-1>", lambda e: self.hw.move_down(False))

        btn_backward = tk.Button(dpad_container, text="BACKWARD", font=dpad_font, bg="#E0E0E0")
        btn_backward.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        btn_backward.bind("<ButtonPress-1>", lambda e: self.hw.move_backward(True))
        btn_backward.bind("<ButtonRelease-1>", lambda e: self.hw.move_backward(False))

        btn_forward = tk.Button(dpad_container, text="FORWARD", font=dpad_font, bg="#E0E0E0")
        btn_forward.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        btn_forward.bind("<ButtonPress-1>", lambda e: self.hw.move_forward(True))
        btn_forward.bind("<ButtonRelease-1>", lambda e: self.hw.move_forward(False))

        # -- Bottom Row: Back Button --
        # Back on Bottom Right
        btn_back = tk.Button(self.motor_frame, text="Back", font=("Arial", 24, "bold"), bg="#A9A9A9",
                             command=self.close_motor_menu, padx=20)
        btn_back.grid(row=2, column=2, sticky="se", padx=40, pady=40)

    def open_motor_menu(self):
        self.video_container.grid_remove()
        self.button_frame.grid_remove()
        self.info_frame.grid_remove()

        self.motor_frame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew")

    def close_motor_menu(self):
        self.motor_frame.grid_remove()

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
        if self.camera.ref_depth is not None:
            self.lbl_cam_height.config(text=f"Camera Height: {self.camera.ref_depth:.3f} m")
        else:
            self.lbl_cam_height.config(text="Camera Height: --")

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
        self.hw.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowcheckGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()