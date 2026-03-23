import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
import numpy as np
import Defect_Masking
import HardwareManager
import AutoLeveler
import Centering
import Default_Workflow
import Shelf_Workflow


class FlowcheckGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FLOWCHECK")
        self.root.configure(bg="white")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.escape_fullscreen)

        self.current_w, self.current_h = 640, 480
        self.last_cw, self.last_ch = 0, 0

        # Start / stop and workflow state
        self.scan_active = False
        self.selected_sku = None
        self.tubs_scanned = 0

        # Debug Mode State
        self.debug_mode = False

        # Drain debug popup
        self.drain_win = None
        self.drain_label = None

        # Flag and frame for custom workflow debug windows (like the shelf arm retraction)
        self.force_drain_window = False
        self.custom_debug_frame = None

        # ==========================================
        # 1. MAIN WINDOW GRID LAYOUT
        # ==========================================
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        # ==========================================
        # 2. TOP ROW: Header Frame
        # ==========================================
        self.header_frame = tk.Frame(self.root, bg="white")
        self.header_frame.grid(row=0, column=0, sticky="nw", padx=40, pady=(20, 10))

        try:
            logo_img = Image.open(os.path.join("files", "flowcheck_logo.png"))
            target_height = 100
            target_width = int(target_height * (logo_img.width / logo_img.height))
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
        # 3. MIDDLE ROW: Video Feed & Right Column
        # ==========================================
        self.video_container = tk.Frame(self.root, bg="white")
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.video_label.bind("<Button-1>", self.on_video_click)
        self.root.bind("<r>", self.on_r_keypress)

        # Master Right Column Frame
        self.right_col_frame = tk.Frame(self.root, bg="white")
        self.right_col_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 40), pady=10)

        # Info Frame (Lives inside Right Column)
        self.info_frame = tk.Frame(self.right_col_frame, bg="white")
        self.info_frame.pack(side="top", anchor="w", fill="x")

        self.info_labels = [
            tk.Label(self.info_frame, text="SKU: None", bg="white", fg="#0044cc"),
            tk.Label(self.info_frame, text="Camera Height: --", bg="white", fg="black"),
            tk.Label(self.info_frame, text="Drain Detected: --", bg="white", fg="black"),
            tk.Label(self.info_frame, text="Defects Detected: --", bg="white", fg="black"),
            tk.Label(self.info_frame, text="Units Scanned: 0", bg="white", fg="black")
        ]

        self.lbl_sku = self.info_labels[0]
        self.lbl_cam_height = self.info_labels[1]
        self.lbl_drain_detected = self.info_labels[2]
        self.lbl_units_scanned = self.info_labels[4]

        for lbl in self.info_labels:
            lbl.pack(anchor="w", pady=6)

        # Right Camera Container (Lives below Info Frame)
        self.right_cam_container = tk.Frame(self.right_col_frame, bg="white")
        self.black_box_1 = tk.Frame(self.right_cam_container, bg="black")
        self.black_box_2 = tk.Frame(self.right_cam_container, bg="black")

        # Labels for the side camera captures
        self.lbl_box_1 = tk.Label(self.black_box_1, bg="black")
        self.lbl_box_1.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_box_2 = tk.Label(self.black_box_2, bg="black")
        self.lbl_box_2.place(relx=0.5, rely=0.5, anchor="center")

        # State variable for freezing the main feed
        self.frozen_main_img = None

        # ==========================================
        # 4. BOTTOM ROW: Main Screen Buttons
        # ==========================================
        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=40, pady=(20, 40))

        for col in range(6):
            self.button_frame.columnconfigure(col, weight=1, uniform="btn_cols")

        self.main_buttons = []

        button_config = [
            ("START", self.toggle_start_stop, "green"),
            ("Select SKU", self.open_sku_menu, "#A9A9A9"),
            ("", None, "#A9A9A9"),
            ("Manual Motor\nControl", self.open_motor_menu, "#A9A9A9"),
            ("Toggle Debug", self.toggle_debug, "#A9A9A9"),
            ("CLOSE", self.on_closing, "orange")
        ]

        for i, (text, cmd, bg_color) in enumerate(button_config):
            btn = tk.Button(
                self.button_frame,
                text=text,
                font=("Arial", 16, "bold"),
                bg=bg_color,
                fg="black",
                height=3,
                command=cmd
            )
            btn.grid(row=0, column=i, sticky="nsew", padx=(0, 15) if i < 5 else 0)
            self.main_buttons.append(btn)

        # ==========================================
        # 5. Initialization & Threading
        # ==========================================
        self.hw = HardwareManager.HardwareManager()

        try:
            self.camera = Defect_Masking.DefectDetector()
        except Exception as e:
            print(f"Camera not detected on startup: {e}")

            class DummyCamera:
                def __init__(self):
                    self.device = None
                    self.ref_depth = None

                def get_frame(self): return None

                def stop(self): pass

                def register_click(self, x, y): pass

                def reset_depth(self): pass

            self.camera = DummyCamera()

        self.no_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "NO CAMERA DETECTED"
        text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(self.no_camera_frame, text, (text_x, text_y), font, 1.2, (255, 255, 255), 3)

        self.leveller = AutoLeveler.AutoLeveler(self.camera.device, self.hw)
        self.drainer = Centering.AutoDrainer(self.camera)
        self.drain_watcher = Centering.DrainWatcher(self.camera)

        # Workflows
        self.strada_workflow = Default_Workflow.StradaWorkflow(self, self.leveller, self.drainer, self.drain_watcher)
        self.shelf_workflow = Shelf_Workflow.ShelfWorkflow(self, self.leveller, self.drainer, self.drain_watcher)

        self.create_sku_menu()
        self.create_motor_menu()

        # Load default SKU & Layout
        self.select_sku("Strada 6030L\n(shower base)")
        self.root.update_idletasks()

        self.latest_frame = None
        self.last_drawn_frame = None
        self.is_running = True

        threading.Thread(target=self.frame_capture_thread, daemon=True).start()
        self.update_video()

    def frame_capture_thread(self):
        while self.is_running:
            try:
                frame = self.camera.get_frame()
                self.latest_frame = frame

                if frame is not None:
                    self.drainer.raw_color = frame
                    self.drain_watcher.raw_color = frame

                    if hasattr(self.camera, "depth_stack") and len(self.camera.depth_stack) > 0:
                        self.drainer.raw_depth = self.camera.depth_stack[0]

            except Exception as e:
                print(f"Camera thread error: {e}")
                self.latest_frame = None

            time.sleep(0.01)

    # ==========================================
    # VIEW TOGGLING HELPER
    # ==========================================
    def toggle_main_view(self, show=True):
        if show:
            self.video_container.grid()
            self.button_frame.grid()
            self.right_col_frame.grid()
        else:
            self.video_container.grid_remove()
            self.button_frame.grid_remove()
            self.right_col_frame.grid_remove()

    # ==========================================
    # SKU SUBMENU LOGIC
    # ==========================================
    def create_sku_menu(self):
        self.sku_frame = tk.Frame(self.root, bg="white")
        for col in range(3):
            self.sku_frame.columnconfigure(col, weight=1, uniform="sku_grid_cols")
        for row in range(2):
            self.sku_frame.rowconfigure(row, weight=1, uniform="sku_grid_rows")

        sku_btns = [
            "Strada 6030L\n(shower base)", "(Skirted Tub)", "(Tub-Shower)",
            "", "*Raw Scan*", "Back"
        ]

        for i, name in enumerate(sku_btns):
            cmd = self.close_sku_menu if name == "Back" else (lambda n=name: self.select_sku(n) if n else None)
            bg_color = "#A9A9A9" if name == "Back" else "#E0E0E0"

            btn = tk.Button(
                self.sku_frame,
                text=name,
                font=("Arial", 24, "bold"),
                bg=bg_color,
                fg="black",
                command=cmd
            )
            btn.grid(row=i // 3, column=i % 3, sticky="nsew", padx=15, pady=15)

    def select_sku(self, sku_name):
        self.selected_sku = sku_name
        self.tubs_scanned = 0

        display_name = sku_name.replace('\n', ' ')
        self.lbl_sku.config(text=f"SKU: {display_name}")
        self.lbl_units_scanned.config(text=f"Units Scanned: {self.tubs_scanned}")

        self.apply_sku_layout(sku_name)
        self.close_sku_menu()

    def apply_sku_layout(self, sku_name):
        # Enforce exactly the same column space for the main window universally
        self.root.columnconfigure(0, weight=3, uniform="expanded_cols")
        self.root.columnconfigure(1, weight=2, uniform="expanded_cols")
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=(10, 10), pady=10)

        if sku_name in ["Strada 6030L\n(shower base)", "(Skirted Tub)"]:
            # Standard padding for main menus
            self.right_col_frame.grid_configure(padx=(10, 40))
            self.right_cam_container.pack_forget()

            for lbl in self.info_labels:
                lbl.config(font=("Arial", 28, "bold"))
            self.lbl_sku.config(font=("Arial", 20, "bold"), fg="#0044cc")

        elif sku_name == "(Tub-Shower)":
            self.right_col_frame.grid_configure(padx=(10, 10))
            self.right_cam_container.pack(side="top", expand=True, fill="both", pady=(5, 0))

            for lbl in self.info_labels:
                lbl.config(font=("Arial", 24, "bold"))
            self.lbl_sku.config(font=("Arial", 18, "bold"), fg="#0044cc")

    def open_sku_menu(self):
        self.toggle_main_view(False)
        self.sku_frame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew", padx=25, pady=(0, 25))

    def close_sku_menu(self):
        self.sku_frame.grid_remove()
        self.toggle_main_view(True)

    # ==========================================
    # MOTOR SUBMENU LOGIC
    # ==========================================
    def create_motor_menu(self):
        self.motor_frame = tk.Frame(self.root, bg="white")
        for i in range(3):
            self.motor_frame.columnconfigure(i, weight=3 if i == 1 else 1)
        for i in range(3):
            self.motor_frame.rowconfigure(i, weight=3 if i == 1 else 1)

        rot_font = ("Arial", 42, "bold")

        btn_cw = tk.Button(self.motor_frame, text="⟳", font=rot_font, bg="#E0E0E0", padx=15, pady=5)
        btn_cw.grid(row=0, column=2, sticky="ne", padx=40, pady=30)
        btn_cw.bind("<ButtonPress-1>", lambda e: self.hw.rotate_cw(True))
        btn_cw.bind("<ButtonRelease-1>", lambda e: self.hw.rotate_cw(False))

        btn_ccw = tk.Button(self.motor_frame, text="⟲", font=rot_font, bg="#E0E0E0", padx=15, pady=5)
        btn_ccw.grid(row=0, column=0, sticky="nw", padx=40, pady=30)
        btn_ccw.bind("<ButtonPress-1>", lambda e: self.hw.rotate_ccw(True))
        btn_ccw.bind("<ButtonRelease-1>", lambda e: self.hw.rotate_ccw(False))

        drain_container = tk.Frame(self.motor_frame, bg="white")
        drain_container.grid(row=0, column=1, sticky="nsew", padx=5, pady=25)
        drain_container.columnconfigure(0, weight=5)
        drain_container.columnconfigure(1, weight=4)
        drain_container.columnconfigure(2, weight=5)
        drain_container.rowconfigure(0, weight=1)

        self.btn_drain = tk.Button(
            drain_container,
            text="CENTER\nDRAIN",
            font=("Arial", 16, "bold"),
            bg="#A9A9A9",
            command=self.toggle_auto_drain
        )
        self.btn_drain.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        dpad = tk.Frame(self.motor_frame, bg="white")
        dpad.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        dpad.columnconfigure(0, weight=5)
        dpad.columnconfigure(1, weight=4)
        dpad.columnconfigure(2, weight=5)
        for i in range(3):
            dpad.rowconfigure(i, weight=1, uniform="dpad_rows")

        dpad_config = [
            ("UP", 0, 1, self.hw.move_up),
            ("DOWN", 2, 1, self.hw.move_down),
            ("BACKWARD", 1, 0, self.hw.move_backward),
            ("FORWARD", 1, 2, self.hw.move_forward)
        ]

        for text, r, c, func in dpad_config:
            btn = tk.Button(dpad, text=text, font=("Arial", 16, "bold"), bg="#E0E0E0")
            btn.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            btn.bind("<ButtonPress-1>", lambda e, f=func: f(True))
            btn.bind("<ButtonRelease-1>", lambda e, f=func: f(False))

        self.btn_auto = tk.Button(
            dpad,
            text="AUTO\nLEVEL",
            font=("Arial", 12, "bold"),
            bg="#A9A9A9",
            command=self.start_auto_level
        )
        self.btn_auto.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        tk.Button(
            self.motor_frame,
            text="Back",
            font=("Arial", 24, "bold"),
            bg="#A9A9A9",
            command=self.close_motor_menu,
            padx=20
        ).grid(row=2, column=2, sticky="se", padx=40, pady=40)

    # --- MAIN MENU LOGIC ---
    def toggle_start_stop(self):
        if self.scan_active:
            self.stop_all_workflows()
            print("Scan Stopped.")
        else:
            if self.selected_sku in ["Strada 6030L\n(shower base)", "(Skirted Tub)"]:
                self.scan_active = True
                self.main_buttons[0].config(text="STOP", bg="red")
                print("Scan Started (Base Workflow).")
                self.strada_workflow.start_workflow()

            elif self.selected_sku == "(Tub-Shower)":
                self.scan_active = True
                self.main_buttons[0].config(text="STOP", bg="red")
                print("Scan Started (Shelf Workflow).")
                self.shelf_workflow.start_workflow()

            else:
                print(f"No workflow assigned for SKU: {self.selected_sku}")
                return

    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.main_buttons[4].config(bg="green")
            if self.drainer.is_running:
                self.open_drain_window()
        else:
            self.main_buttons[4].config(bg="#A9A9A9")
            self.close_drain_window()

    def stop_all_workflows(self):
        self.scan_active = False
        self.main_buttons[0].config(text="START", bg="green")

        self.strada_workflow.stop_workflow()
        self.shelf_workflow.stop_workflow()

        self.btn_auto.config(bg="#A9A9A9")
        self.btn_drain.config(bg="#A9A9A9")
        self.close_drain_window()
        self.reset_drain_status()

    def increment_tub_count(self):
        def _update():
            self.tubs_scanned += 1
            self.lbl_units_scanned.config(text=f"Units Scanned: {self.tubs_scanned}")

        self.root.after(0, _update)

    # --- IMAGE FREEZING & BOX POPULATION ---
    def set_frozen_main_image(self, img):
        self.frozen_main_img = img

    def clear_frozen_main_image(self):
        self.frozen_main_img = None

    def set_left_box_image(self, img):
        self._set_box_image(self.lbl_box_1, self.black_box_1, img)

    def set_right_box_image(self, img):
        self._set_box_image(self.lbl_box_2, self.black_box_2, img)

    def _set_box_image(self, label, container, img):
        cw = container.winfo_width()
        ch = container.winfo_height()
        if cw > 10 and ch > 10 and img is not None:
            img_resized = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
            label.imgtk = imgtk
            label.configure(image=imgtk)

    def clear_box_images(self):
        self.lbl_box_1.configure(image='')
        self.lbl_box_1.imgtk = None
        self.lbl_box_2.configure(image='')
        self.lbl_box_2.imgtk = None

    # --- AUTO LEVEL ---
    def start_auto_level(self):
        self.leveller.start()

    # --- AUTO DRAIN ---
    def toggle_auto_drain(self):
        if self.drainer.is_running:
            self.drainer.stop()
        else:
            self.btn_drain.config(bg="green")
            self.open_drain_window()
            self.drainer.start(
                callback=self.auto_drain_done,
                stop_on_first_drain=False,
            )

    def open_drain_window(self):
        if not self.debug_mode:
            return

        if self.drain_win is not None:
            try:
                if self.drain_win.winfo_exists():
                    self.drain_win.deiconify()
                    self.drain_win.lift()
                    return
            except Exception:
                pass

        self.drain_win = tk.Toplevel(self.root)
        self.drain_win.title("Auto Drain Tracking")
        self.drain_win.geometry("640x480")
        self.drain_win.configure(bg="black")
        self.drain_win.protocol("WM_DELETE_WINDOW", self.close_drain_window)

        self.drain_label = tk.Label(self.drain_win, bg="black")
        self.drain_label.pack(expand=True, fill="both")

        self.update_drain_window()

    def close_drain_window(self):
        if self.drain_win is not None:
            try:
                if self.drain_win.winfo_exists():
                    self.drain_win.destroy()
            except Exception:
                pass

        self.drain_win = None
        self.drain_label = None

    def update_drain_window(self):
        # Allow workflow to keep window open via force flag
        force_open = getattr(self, "force_drain_window", False)

        if not self.drainer.is_running and not force_open:
            self.close_drain_window()
            return

        # Prioritize the drainer's frame if it's running, otherwise use custom workflow frame
        if self.drainer.is_running:
            frame = getattr(self.drainer, "display_frame", None)
        else:
            frame = getattr(self, "custom_debug_frame", None)

        if frame is not None and self.drain_label is not None:
            try:
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv2_img)
                imgtk = ImageTk.PhotoImage(image=pil_img)

                self.drain_label.imgtk = imgtk
                self.drain_label.configure(image=imgtk)
            except Exception as e:
                print(f"Drain Popup Error: {e}")

        self.root.after(30, self.update_drain_window)

    def auto_drain_done(self, status):
        def _finish():
            self.btn_drain.config(bg="#A9A9A9")
            self.close_drain_window()

        self.root.after(0, _finish)

    def on_drain_status_changed(self, detected):
        def _update():
            if not self.scan_active:
                self.lbl_drain_detected.config(text="Drain Detected: --", fg="black")
            elif detected:
                self.lbl_drain_detected.config(text="Drain Detected: Yes", fg="green")
            else:
                self.lbl_drain_detected.config(text="Drain Detected: No", fg="black")

        self.root.after(0, _update)

    def reset_drain_status(self):
        def _update():
            self.lbl_drain_detected.config(text="Drain Detected: --", fg="black")

        self.root.after(0, _update)

    # --- MENU NAVIGATION ---
    def open_motor_menu(self):
        self.toggle_main_view(False)
        self.motor_frame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew")

    def close_motor_menu(self):
        self.stop_all_workflows()
        self.motor_frame.grid_remove()
        self.toggle_main_view(True)

    # ==========================================
    # CORE FUNCTIONS
    # ==========================================
    def escape_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

    def on_video_click(self, event):
        scale_x = 640 / self.current_w
        scale_y = 480 / self.current_h
        self.camera.register_click(int(event.x * scale_x), int(event.y * scale_y))

    def on_r_keypress(self, event):
        self.camera.reset_depth()

    def update_video(self):
        # 1. LIVE HEIGHT UPDATE
        if hasattr(self.camera, "depth_stack") and len(self.camera.depth_stack) > 0:
            depth_img = self.camera.depth_stack[0]
            dist, _ = Centering.get_basin_distance(depth_img)
            if dist is not None:
                self.lbl_cam_height.config(text=f"Camera Height: {dist:.3f} m")
            else:
                self.lbl_cam_height.config(text="Camera Height: --")
        else:
            self.lbl_cam_height.config(text="Camera Height: --")

        # 2. IMAGE DRAWING & FREEZE LOGIC
        # Show the frozen image if one exists, otherwise stream live video
        frame = self.frozen_main_img if self.frozen_main_img is not None else (
            self.latest_frame if self.latest_frame is not None else self.no_camera_frame)

        if frame is not self.last_drawn_frame:
            self.last_drawn_frame = frame

            cw = self.video_container.winfo_width()
            ch = self.video_container.winfo_height()

            if cw > 10 and ch > 10:
                if cw != self.last_cw or ch != self.last_ch:
                    aspect = 640 / 480
                    container_aspect = cw / ch

                    if container_aspect > aspect:
                        self.current_h, self.current_w = ch, int(ch * aspect)
                    else:
                        self.current_w, self.current_h = cw, int(cw / aspect)

                    self.last_cw, self.last_ch = cw, ch

                if self.selected_sku == "(Tub-Shower)":
                    rc_w = self.right_cam_container.winfo_width()
                    rc_h = self.right_cam_container.winfo_height()

                    if rc_w > 10 and rc_h > 10:
                        gap = 10
                        box_h = rc_h
                        box_w = int(box_h * (640 / 480))

                        if (2 * box_w + gap) > rc_w:
                            box_w = (rc_w - gap) // 2
                            box_h = int(box_w * (480 / 640))

                        x_offset = (rc_w - (2 * box_w + gap)) // 2
                        y_offset = (rc_h - box_h) // 2

                        self.black_box_1.place(x=x_offset, y=y_offset, width=box_w, height=box_h)
                        self.black_box_2.place(x=x_offset + box_w + gap, y=y_offset, width=box_w, height=box_h)

                frame_resized = cv2.resize(frame, (self.current_w, self.current_h), interpolation=cv2.INTER_LINEAR)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def on_closing(self):
        self.is_running = False
        self.stop_all_workflows()
        self.leveller.shutdown()
        if hasattr(self.camera, "stop"):
            self.camera.stop()
        self.hw.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowcheckGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()