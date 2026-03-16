import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import Defect_Masking  # Imports your camera script


class FlowcheckGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FLOWCHECK")
        self.root.configure(bg="white")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.escape_fullscreen)

        # Track current display size to map mouse clicks back to native 640x480
        self.current_w = 640
        self.current_h = 480

        # ==========================================
        # 1. MAIN WINDOW GRID LAYOUT
        # ==========================================
        self.root.columnconfigure(0, weight=1, uniform="main_cols")
        self.root.columnconfigure(1, weight=1, uniform="main_cols")

        self.root.rowconfigure(0, weight=0)  # Row 0: Top Header area
        self.root.rowconfigure(1, weight=1)  # Row 1: Middle area
        self.root.rowconfigure(2, weight=0)  # Row 2: Bottom area

        # ==========================================
        # 2. TOP ROW: Header Frame (Logo + Version) - STAYS VISIBLE
        # ==========================================
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

        # ==========================================
        # 3. MIDDLE ROW: Video Feed
        # ==========================================
        self.video_container = tk.Frame(self.root, bg="white")
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=(40, 20), pady=10)

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

        # Hooked up the command for "Select SKU"
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
        # 5. Build Submenus
        # ==========================================
        self.create_sku_menu()

        self.root.update_idletasks()

        # ==========================================
        # 6. Initialize Camera
        # ==========================================
        self.camera = Defect_Masking.DefectDetector()
        self.update_video()

    # ==========================================
    # SUBMENU LOGIC
    # ==========================================
    def create_sku_menu(self):
        """Builds the hidden 3x2 SKU selection layout."""
        self.sku_frame = tk.Frame(self.root, bg="white")

        # Create a 3x2 grid inside the sku_frame
        for col in range(3):
            self.sku_frame.columnconfigure(col, weight=1, uniform="sku_grid_cols")
        for row in range(2):
            self.sku_frame.rowconfigure(row, weight=1, uniform="sku_grid_rows")

        # Generate the 6 large buttons
        for r in range(2):
            for c in range(3):
                if r == 1 and c == 2:
                    # Bottom Right Button = Back
                    btn = tk.Button(
                        self.sku_frame,
                        text="Back",
                        font=("Arial", 24, "bold"),
                        bg="#A9A9A9",
                        command=self.close_sku_menu
                    )
                else:
                    # The other 5 blank buttons
                    btn = tk.Button(
                        self.sku_frame,
                        text="",
                        font=("Arial", 24, "bold"),
                        bg="#E0E0E0"
                    )

                # Add padding around the buttons so they don't touch
                btn.grid(row=r, column=c, sticky="nsew", padx=15, pady=15)

    def open_sku_menu(self):
        """Hides the main screen elements and shows the SKU menu."""
        self.video_container.grid_remove()
        self.button_frame.grid_remove()

        # Place the SKU menu across the remaining space (Rows 1 & 2)
        self.sku_frame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew", padx=25, pady=(0, 25))

    def close_sku_menu(self):
        """Hides the SKU menu and brings back the main screen."""
        self.sku_frame.grid_remove()

        self.video_container.grid()
        self.button_frame.grid()

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
        frame = self.camera.get_frame()

        if frame is not None:
            cw = self.video_container.winfo_width()
            ch = self.video_container.winfo_height()

            if cw > 10 and ch > 10:
                aspect = 640 / 480
                container_aspect = cw / ch

                if container_aspect > aspect:
                    self.current_h = ch
                    self.current_w = int(ch * aspect)
                else:
                    self.current_w = cw
                    self.current_h = int(cw / aspect)

                frame = cv2.resize(frame, (self.current_w, self.current_h), interpolation=cv2.INTER_LINEAR)

            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img)
            imgtk = ImageTk.PhotoImage(image=pil_img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_video)

    def on_closing(self):
        self.camera.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowcheckGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()