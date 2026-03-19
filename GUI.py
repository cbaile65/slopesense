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
        self.workflow_thread = None
        self.workflow_id = 0

        # Drain debug popup
        self.drain_win = None
        self.drain_label = None

        # Strada assembly-line state
        self.strada_cycle_thread = None
        self.strada_cycle_running = False
        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.center_abort_requested = False
        self.last_drain_seen_state = False
        self.autolevel_done_for_current_tub = False

        # ==========================================
        # 1. MAIN WINDOW GRID LAYOUT
        # ==========================================
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        # ==========================================
        # 2. TOP ROW: Header Frame & Info Frame
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

        self.info_frame = tk.Frame(self.root, bg="white")

        self.info_labels = [
            tk.Label(self.info_frame, text="Camera Height: --", bg="white", fg="black"),
            tk.Label(self.info_frame, text="Drain Detected: --", bg="white", fg="black"),
            tk.Label(self.info_frame, text="Defects Detected: --", bg="white", fg="black"),
            tk.Label(self.info_frame, text="Units Scanned: --", bg="white", fg="black")
        ]

        self.lbl_cam_height = self.info_labels[0]
        self.lbl_drain_detected = self.info_labels[1]

        for lbl in self.info_labels:
            lbl.pack(anchor="w", pady=6)

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

        self.main_buttons = []

        button_config = [
            ("START", self.toggle_start_stop, "green"),
            ("Select SKU", self.open_sku_menu, "#A9A9A9"),
            ("", None, "#A9A9A9"),
            ("Manual Motor\nControl", self.open_motor_menu, "#A9A9A9"),
            ("Toggle Debug", None, "#A9A9A9"),
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
        self.camera = Defect_Masking.DefectDetector()

        self.leveller = AutoLeveler.AutoLeveler(self.camera.device, self.hw)
        self.drainer = Centering.AutoDrainer(self.camera)
        self.drain_watcher = Centering.DrainWatcher(self.camera)

        self.create_sku_menu()
        self.create_motor_menu()

        self.apply_sku_layout("Strada\n(Shower Base)")
        self.selected_sku = "Strada\n(Shower Base)"
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
                if frame is not None:
                    self.latest_frame = frame

                    # Feed current image into both centering + watcher
                    self.drainer.raw_color = frame
                    self.drain_watcher.raw_color = frame

                    if hasattr(self.camera, "depth_stack"):
                        self.drainer.raw_depth = self.camera.depth_stack[0]

            except Exception as e:
                print(f"Camera thread error: {e}")

            time.sleep(0.01)

    # ==========================================
    # VIEW TOGGLING HELPER
    # ==========================================
    def toggle_main_view(self, show=True):
        if show:
            self.video_container.grid()
            self.button_frame.grid()
            self.info_frame.grid()
        else:
            self.video_container.grid_remove()
            self.button_frame.grid_remove()
            self.info_frame.grid_remove()

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
            "Strada\n(Shower Base)", "(Skirted Tub)", "(Tub-Shower)",
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
        self.apply_sku_layout(sku_name)
        self.close_sku_menu()

    def apply_sku_layout(self, sku_name):
        if sku_name in ["Strada\n(Shower Base)", "(Skirted Tub)"]:
            self.root.columnconfigure(0, weight=3, uniform="expanded_cols")
            self.root.columnconfigure(1, weight=2, uniform="expanded_cols")
            self.video_container.grid(row=1, column=0, sticky="nsew", padx=(10, 10), pady=10)
            self.info_frame.grid_configure(row=1, column=1, rowspan=1, sticky="w", padx=(10, 20), pady=0)

            for lbl in self.info_labels:
                lbl.config(font=("Arial", 28, "bold"))

        elif sku_name == "(Tub-Shower)":
            self.root.columnconfigure(0, weight=1, uniform="main_cols")
            self.root.columnconfigure(1, weight=1, uniform="main_cols")
            self.video_container.grid(row=1, column=0, sticky="nsew", padx=(40, 20), pady=10)
            self.info_frame.grid_configure(row=0, column=1, rowspan=2, sticky="nw", padx=(20, 40), pady=(30, 10))

            for lbl in self.info_labels:
                lbl.config(font=("Arial", 24, "bold"))

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
            self.scan_active = True
            self.main_buttons[0].config(text="STOP", bg="red")
            print("Scan Started.")
            self.start_selected_sku_workflow()

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
        if not self.drainer.is_running:
            self.close_drain_window()
            return

        frame = getattr(self.drainer, "display_frame", None)

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

    # ==========================================
    # STRADA ASSEMBLY-LINE WORKFLOW
    # ==========================================
    def start_selected_sku_workflow(self):
        if self.selected_sku != "Strada\n(Shower Base)":
            print(f"No workflow assigned for SKU: {self.selected_sku}")
            return

        self.workflow_id += 1
        current_workflow_id = self.workflow_id

        self.strada_cycle_running = False
        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.center_abort_requested = False
        self.last_drain_seen_state = False
        self.autolevel_done_for_current_tub = False

        self.on_drain_status_changed(False)

        # Always run passive drain watcher while START is active
        self.drain_watcher.start(
            status_callback=self.on_drain_status_changed
        )

        def _workflow_manager():
            print("[Strada Workflow] Running: autolevel first, then wait for drain.")
            try:
                while self.scan_active and self.workflow_id == current_workflow_id:
                    drain_seen = bool(self.drain_watcher.drain_present)
                    drain_locked = bool(self.drain_watcher.locked_drain_ready)

                    # Drain lost -> tub is gone -> re-arm full cycle
                    if not drain_seen and self.last_drain_seen_state:
                        print("[Strada Workflow] Drain disappeared. Re-arming next cycle.")
                        self.current_tub_has_drain = False
                        self.current_tub_processed = False
                        self.autolevel_done_for_current_tub = False

                        if self.strada_cycle_running and self.drainer.is_running:
                            print("[Strada Workflow] Drain lost during centering. Aborting cycle.")
                            self.center_abort_requested = True
                            self.drainer.stop()
                            self.root.after(0, self.close_drain_window)

                    # Run autolevel first for each new tub before drain search matters
                    if (
                        not self.autolevel_done_for_current_tub
                        and not self.strada_cycle_running
                    ):
                        print("[Strada Workflow] Starting autolevel for next tub.")
                        self.start_strada_cycle(current_workflow_id, mode="autolevel_then_wait_for_drain")

                    # After autolevel, allow either:
                    # 1) a fresh new lock event, OR
                    # 2) an already-held locked drain
                    elif (
                        self.autolevel_done_for_current_tub
                        and not self.current_tub_processed
                        and not self.strada_cycle_running
                    ):
                        fresh_lock = self.drain_watcher.consume_new_lock_event()

                        if fresh_lock or drain_locked:
                            print("[Strada Workflow] Drain ready. Starting centering cycle.")
                            self.current_tub_has_drain = True
                            self.start_strada_cycle(current_workflow_id, mode="center_only")

                    self.last_drain_seen_state = drain_seen
                    time.sleep(0.05)

            except Exception as e:
                print(f"[Workflow Manager Error] {e}")
                self.root.after(0, self.stop_all_workflows)

        self.workflow_thread = threading.Thread(target=_workflow_manager, daemon=True)
        self.workflow_thread.start()

    def start_strada_cycle(self, workflow_id_at_start, mode):
        if self.strada_cycle_running:
            return

        self.strada_cycle_running = True
        self.center_abort_requested = False

        def _cycle():
            try:
                if not self.scan_active or self.workflow_id != workflow_id_at_start:
                    return

                # ------------------------------------------
                # MODE 1: autolevel first, then return to watcher
                # ------------------------------------------
                if mode == "autolevel_then_wait_for_drain":
                    print("[Strada Cycle] Starting autolevel.")
                    self.root.after(0, lambda: self.btn_auto.config(bg="green"))
                    self.leveller.start()

                    while (
                        self.scan_active
                        and self.workflow_id == workflow_id_at_start
                        and self.leveller.is_running
                    ):
                        time.sleep(0.05)

                    self.root.after(0, lambda: self.btn_auto.config(bg="#A9A9A9"))

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    self.autolevel_done_for_current_tub = True
                    print("[Strada Cycle] Autolevel complete. Waiting for drain.")

                # ------------------------------------------
                # MODE 2: drain already found, do center only
                # ------------------------------------------
                elif mode == "center_only":
                    if not self.drain_watcher.drain_present:
                        print("[Strada Cycle] Drain not present anymore. Canceling centering.")
                        return

                    print("[Strada Cycle] Starting autocenter.")
                    self.root.after(0, lambda: self.btn_drain.config(bg="green"))
                    self.root.after(0, self.open_drain_window)

                    self.drainer.start(
                        callback=self.auto_drain_done,
                        stop_on_first_drain=False,
                    )

                    while (
                        self.scan_active
                        and self.workflow_id == workflow_id_at_start
                        and self.drainer.is_running
                    ):
                        if not self.drain_watcher.drain_present:
                            print("[Strada Cycle] Drain lost during autocenter. Stopping centering.")
                            self.center_abort_requested = True
                            self.drainer.stop()
                            self.root.after(0, self.close_drain_window)
                            break

                        time.sleep(0.05)

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    if self.center_abort_requested:
                        print("[Strada Cycle] Cycle aborted. Waiting for drain loss / next tub.")
                        return

                    print("[Strada Cycle] Autocenter complete. Tub marked processed.")
                    self.current_tub_processed = True

            except Exception as e:
                print(f"[Strada Cycle Error] {e}")
                self.root.after(0, self.stop_all_workflows)

            finally:
                self.root.after(0, lambda: self.btn_auto.config(bg="#A9A9A9"))
                self.root.after(0, lambda: self.btn_drain.config(bg="#A9A9A9"))
                self.root.after(0, self.close_drain_window)
                self.strada_cycle_running = False

        self.strada_cycle_thread = threading.Thread(target=_cycle, daemon=True)
        self.strada_cycle_thread.start()

    def stop_all_workflows(self):
        self.workflow_id += 1
        self.scan_active = False
        self.main_buttons[0].config(text="START", bg="green")

        self.strada_cycle_running = False
        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.center_abort_requested = False
        self.last_drain_seen_state = False
        self.autolevel_done_for_current_tub = False

        self.leveller.stop()
        self.drainer.stop()
        self.drain_watcher.stop()

        self.btn_auto.config(bg="#A9A9A9")
        self.btn_drain.config(bg="#A9A9A9")
        self.close_drain_window()
        self.reset_drain_status()

    # --- MENU NAVIGATION ---
    def open_motor_menu(self):
        self.toggle_main_view(False)
        self.motor_frame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew")

    def close_motor_menu(self):
        self.leveller.stop()
        self.drainer.stop()
        self.drain_watcher.stop()
        self.btn_auto.config(bg="#A9A9A9")
        self.btn_drain.config(bg="#A9A9A9")
        self.close_drain_window()
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
                        self.current_h, self.current_w = ch, int(ch * aspect)
                    else:
                        self.current_w, self.current_h = cw, int(cw / aspect)

                    self.last_cw, self.last_ch = cw, ch

                frame_resized = cv2.resize(frame, (self.current_w, self.current_h), interpolation=cv2.INTER_LINEAR)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)

    def on_closing(self):
        self.is_running = False
        self.stop_all_workflows()
        self.leveller.shutdown()
        self.camera.stop()
        self.hw.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowcheckGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()