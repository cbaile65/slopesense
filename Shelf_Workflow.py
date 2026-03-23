import threading
import time
import cv2
import numpy as np
from Centering import get_basin_distance, TARGET_DISTANCE_M

# =========================================================
# WORKFLOW SETTINGS
# =========================================================
DRAIN_LOSS_DEBOUNCE_SEC = 4.0  # Seconds the drain must be missing to trigger a reset


class ShelfWorkflow:
    def __init__(self, gui_app, leveller, drainer, drain_watcher):
        self.gui = gui_app
        self.leveller = leveller
        self.drainer = drainer
        self.drain_watcher = drain_watcher

        self.workflow_id = 0
        self.scan_active = False
        self.workflow_thread = None

        self.strada_cycle_thread = None
        self.strada_cycle_running = False

        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.autolevel_done_for_current_tub = False

        # State variables for scanning and debouncing
        self.scanning_in_progress = False
        self.drain_lost_timestamp = None

    def start_workflow(self):
        self.scan_active = True
        self.workflow_id += 1
        current_workflow_id = self.workflow_id

        self.strada_cycle_running = False
        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.autolevel_done_for_current_tub = False
        self.scanning_in_progress = False
        self.drain_lost_timestamp = None

        self.gui.on_drain_status_changed(False)

        # Clear frozen frames and side boxes from the previous tub (Thread-Safe)
        if hasattr(self.gui, "clear_frozen_main_image"):
            self.gui.root.after(0, self.gui.clear_frozen_main_image)
            self.gui.root.after(0, self.gui.clear_box_images)

        # Always run passive drain watcher while START is active
        self.drain_watcher.start(status_callback=self.gui.on_drain_status_changed)

        def _workflow_manager():
            print("[Shelf Workflow] Running: autolevel, wait for drain, center, then scan shelves.")
            try:
                while self.scan_active and self.workflow_id == current_workflow_id:
                    drain_seen = bool(self.drain_watcher.drain_present)
                    drain_locked = bool(self.drain_watcher.locked_drain_ready)

                    # --- DRAIN LOSS DEBOUNCE LOGIC ---
                    if not self.scanning_in_progress:
                        if not drain_seen:
                            if self.drain_lost_timestamp is None:
                                self.drain_lost_timestamp = time.time()
                            elif time.time() - self.drain_lost_timestamp >= DRAIN_LOSS_DEBOUNCE_SEC:
                                print("[Shelf Workflow] Drain fully lost. Re-arming next cycle.")

                                if self.current_tub_processed:
                                    print(
                                        "[Shelf Workflow] Tub successfully processed and removed. Incrementing count.")
                                    self.gui.increment_tub_count()

                                self.current_tub_has_drain = False
                                self.current_tub_processed = False
                                self.autolevel_done_for_current_tub = False
                                self.drain_lost_timestamp = None

                                # Safely pass the clear commands to the main UI thread
                                if hasattr(self.gui, "clear_frozen_main_image"):
                                    self.gui.root.after(0, self.gui.clear_frozen_main_image)
                                    self.gui.root.after(0, self.gui.clear_box_images)

                                if self.strada_cycle_running and self.drainer.is_running:
                                    print("[Shelf Workflow] Drain lost during centering. Aborting cycle.")
                                    self.drainer.stop()
                                    self.gui.root.after(0, self.gui.close_drain_window)
                        else:
                            self.drain_lost_timestamp = None

                    # Run autolevel first for each new tub
                    if not self.autolevel_done_for_current_tub and not self.strada_cycle_running:
                        print("[Shelf Workflow] Starting autolevel for next tub.")
                        self.start_shelf_cycle(current_workflow_id, mode="autolevel_then_wait_for_drain")

                    # After autolevel, proceed to centering AND shelf scanning
                    elif self.autolevel_done_for_current_tub and not self.current_tub_processed and not self.strada_cycle_running:
                        fresh_lock = self.drain_watcher.consume_new_lock_event()

                        if fresh_lock or drain_locked:
                            print("[Shelf Workflow] Drain ready. Starting center & shelf scan cycle.")
                            self.current_tub_has_drain = True
                            self.start_shelf_cycle(current_workflow_id, mode="center_and_scan")

                    time.sleep(0.05)

            except Exception as e:
                print(f"[Shelf Workflow Manager Error] {e}")
                self.gui.root.after(0, self.gui.stop_all_workflows)

        self.workflow_thread = threading.Thread(target=_workflow_manager, daemon=True)
        self.workflow_thread.start()

    def start_shelf_cycle(self, workflow_id_at_start, mode):
        if self.strada_cycle_running:
            return

        self.strada_cycle_running = True

        def _cycle():
            try:
                if not self.scan_active or self.workflow_id != workflow_id_at_start:
                    return

                # ------------------------------------------
                # MODE 1: autolevel first, then return to watcher
                # ------------------------------------------
                if mode == "autolevel_then_wait_for_drain":
                    print("[Shelf Cycle] Starting autolevel.")
                    self.gui.root.after(0, lambda: self.gui.btn_auto.config(bg="green"))
                    self.leveller.start()

                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.leveller.is_running:
                        time.sleep(0.05)

                    self.gui.root.after(0, lambda: self.gui.btn_auto.config(bg="#A9A9A9"))

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    self.autolevel_done_for_current_tub = True
                    print("[Shelf Cycle] Autolevel complete. Waiting for drain.")

                # ------------------------------------------
                # MODE 2: Center, freeze image, move up, scan shelves, retract, move down
                # ------------------------------------------
                elif mode == "center_and_scan":
                    if not self.drain_watcher.drain_present:
                        print("[Shelf Cycle] Drain not present anymore. Canceling centering.")
                        return

                    print("[Shelf Cycle] Starting autocenter.")
                    self.gui.root.after(0, lambda: self.gui.btn_drain.config(bg="green"))
                    self.gui.root.after(0, self.gui.open_drain_window)

                    self.drainer.start(
                        callback=self.gui.auto_drain_done,
                        stop_on_first_drain=False,
                    )

                    cycle_aborted = False
                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.drainer.is_running:
                        if not self.drain_watcher.drain_present:
                            print("[Shelf Cycle] Drain lost during autocenter. Stopping cycle.")
                            cycle_aborted = True
                            self.drainer.stop()
                            self.gui.root.after(0, self.gui.close_drain_window)
                            break
                        time.sleep(0.05)

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    if cycle_aborted:
                        print("[Shelf Cycle] Cycle aborted. Waiting for drain loss / next tub.")
                        return

                    print("[Shelf Cycle] Autocenter complete. Initiating Shelf Capture Sequence.")

                    # Tell the workflow manager to ignore drain loss while we do the aerial maneuvers
                    self.scanning_in_progress = True

                    # --- PHASE 1: FREEZE BASE IMAGE ---
                    print("[Shelf Cycle] Waiting 3 seconds before capturing base image...")
                    time.sleep(3.0)

                    base_img = self.gui.latest_frame.copy() if self.gui.latest_frame is not None else None
                    if base_img is not None and hasattr(self.gui, "set_frozen_main_image"):
                        self.gui.root.after(0, self.gui.set_frozen_main_image, base_img)

                    # --- PHASE 2: MOVE UP TO 1.60M (+/- 10cm threshold) ---
                    print("[Shelf Cycle] Moving up to 1.60m...")
                    target_up_min = 1.50
                    target_up_max = 1.70

                    while self.scan_active and self.workflow_id == workflow_id_at_start:
                        depth_img = self.gui.camera.depth_stack[0] if hasattr(self.gui.camera, "depth_stack") and len(
                            self.gui.camera.depth_stack) > 0 else None
                        if depth_img is not None:
                            dist, _ = get_basin_distance(depth_img)
                            if dist is not None:
                                if dist < target_up_min:
                                    self.gui.hw.move_down(False)
                                    self.gui.hw.move_up(True)
                                elif dist > target_up_max:
                                    self.gui.hw.move_up(False)
                                    self.gui.hw.move_down(True)
                                else:
                                    print(f"[Shelf Cycle] Reached target height: {dist:.3f}m")
                                    self.gui.hw.move_up(False)
                                    self.gui.hw.move_down(False)
                                    break
                        time.sleep(0.05)

                    self.gui.hw.move_up(False)
                    self.gui.hw.move_down(False)
                    if not self.scan_active or self.workflow_id != workflow_id_at_start: return

                    # --- PHASE 3: TURN CW (45 DEGREES) & CAPTURE LEFT BOX ---
                    print("[Shelf Cycle] Turning CW to 45 degrees for Left Box...")
                    target_left = 45
                    self.gui.hw.rotate_cw(True)
                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.gui.hw.servo_angle > target_left:
                        time.sleep(0.02)
                    self.gui.hw.rotate_cw(False)

                    print("[Shelf Cycle] Waiting 3 seconds before capturing left shelf...")
                    time.sleep(3.0)

                    left_img = self.gui.latest_frame.copy() if self.gui.latest_frame is not None else None
                    if left_img is not None and hasattr(self.gui, "set_left_box_image"):
                        self.gui.root.after(0, self.gui.set_left_box_image, left_img)

                    # --- PHASE 4: TURN CCW (135 DEGREES) & CAPTURE RIGHT BOX ---
                    print("[Shelf Cycle] Turning CCW to 135 degrees for Right Box...")
                    target_right = 135
                    self.gui.hw.rotate_ccw(True)
                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.gui.hw.servo_angle < target_right:
                        time.sleep(0.02)
                    self.gui.hw.rotate_ccw(False)

                    print("[Shelf Cycle] Waiting 3 seconds before capturing right shelf...")
                    time.sleep(3.0)

                    right_img = self.gui.latest_frame.copy() if self.gui.latest_frame is not None else None
                    if right_img is not None and hasattr(self.gui, "set_right_box_image"):
                        self.gui.root.after(0, self.gui.set_right_box_image, right_img)

                    # --- PHASE 5: RETURN SERVO TO CENTER (90 DEGREES) ---
                    print("[Shelf Cycle] Resetting camera to center (90 degrees)...")
                    target_center = 90
                    self.gui.hw.rotate_cw(True)  # We are at 135, so move CW to get to 90
                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.gui.hw.servo_angle > target_center:
                        time.sleep(0.02)
                    self.gui.hw.rotate_cw(False)

                    # --- PHASE 6: RETRACT UNTIL DRAIN IS AT TOP EDGE ---
                    print("[Shelf Cycle] Retracting arm. Waiting for drain to hit top edge of FOV...")
                    TOP_Y_MIN, TOP_Y_MAX = 20, 100

                    # Open debug window manually if toggle is active
                    if self.gui.debug_mode:
                        self.gui.force_drain_window = True
                        self.gui.root.after(0, self.gui.open_drain_window)

                    while self.scan_active and self.workflow_id == workflow_id_at_start:
                        drain_pos = self.drain_watcher.locked_drain if self.drain_watcher.locked_drain_ready else None

                        # -- Generate Custom Debug Frame for Phase 6 --
                        if self.gui.debug_mode and getattr(self.gui, "latest_frame", None) is not None:
                            debug_img = self.gui.latest_frame.copy()
                            # Draw Target Zone
                            cv2.rectangle(debug_img, (0, TOP_Y_MIN), (640, TOP_Y_MAX), (0, 255, 255), 2)
                            cv2.putText(debug_img, "TARGET EDGE ZONE", (10, TOP_Y_MAX - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 255), 2)

                            if drain_pos:
                                dx, dy, dr = drain_pos
                                color = (0, 255, 0) if TOP_Y_MIN <= dy <= TOP_Y_MAX else (0, 0, 255)
                                cv2.circle(debug_img, (int(dx), int(dy)), int(dr), color, 3)
                                cv2.putText(debug_img, f"Drain Y: {int(dy)}", (int(dx) - 40, int(dy) - dr - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            else:
                                cv2.putText(debug_img, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2)

                            self.gui.custom_debug_frame = debug_img

                        # --- Movement Logic ---
                        if not drain_pos:
                            self.gui.hw.move_forward(False)
                            self.gui.hw.move_backward(False)
                            time.sleep(0.05)
                            continue

                        x, y, r = drain_pos

                        if TOP_Y_MIN <= y <= TOP_Y_MAX:
                            print(f"[Shelf Cycle] Arm retracted. Drain is at Top Edge (Y: {y})")
                            self.gui.hw.move_forward(False)
                            self.gui.hw.move_backward(False)
                            break

                        if y > TOP_Y_MAX:
                            self.gui.hw.move_forward(False)
                            self.gui.hw.move_backward(True)
                        elif y < TOP_Y_MIN:
                            self.gui.hw.move_backward(False)
                            self.gui.hw.move_forward(True)

                        time.sleep(0.05)

                    self.gui.hw.move_forward(False)
                    self.gui.hw.move_backward(False)

                    # Close debug window
                    self.gui.force_drain_window = False
                    self.gui.root.after(0, self.gui.close_drain_window)

                    # --- PHASE 7: MOVE BACK DOWN TO 1.25M ---
                    print("[Shelf Cycle] Moving back down to 1.25m...")
                    target_down_min = 1.15
                    target_down_max = 1.35

                    while self.scan_active and self.workflow_id == workflow_id_at_start:
                        depth_img = self.gui.camera.depth_stack[0] if hasattr(self.gui.camera, "depth_stack") and len(
                            self.gui.camera.depth_stack) > 0 else None
                        if depth_img is not None:
                            dist, _ = get_basin_distance(depth_img)
                            if dist is not None:
                                if dist > target_down_max:
                                    self.gui.hw.move_up(False)
                                    self.gui.hw.move_down(True)
                                elif dist < target_down_min:
                                    self.gui.hw.move_down(False)
                                    self.gui.hw.move_up(True)
                                else:
                                    print(f"[Shelf Cycle] Returned to base height: {dist:.3f}m")
                                    self.gui.hw.move_up(False)
                                    self.gui.hw.move_down(False)
                                    break
                        time.sleep(0.05)

                    self.gui.hw.move_up(False)
                    self.gui.hw.move_down(False)

                    print("[Shelf Cycle] Shelf sequence complete. Waiting for tub removal.")
                    self.current_tub_processed = True
                    self.scanning_in_progress = False  # Re-enable drain loss checking

            except Exception as e:
                print(f"[Shelf Cycle Error] {e}")
                try:
                    # Failsafe hardware stops if the thread crashes
                    self.gui.hw.move_up(False)
                    self.gui.hw.move_down(False)
                    self.gui.hw.move_forward(False)
                    self.gui.hw.move_backward(False)
                    self.gui.hw.rotate_cw(False)
                    self.gui.hw.rotate_ccw(False)
                except Exception:
                    pass
                self.gui.root.after(0, self.gui.stop_all_workflows)

            finally:
                self.gui.force_drain_window = False
                self.gui.root.after(0, lambda: self.gui.btn_auto.config(bg="#A9A9A9"))
                self.gui.root.after(0, lambda: self.gui.btn_drain.config(bg="#A9A9A9"))
                self.gui.root.after(0, self.gui.close_drain_window)
                self.strada_cycle_running = False

        self.strada_cycle_thread = threading.Thread(target=_cycle, daemon=True)
        self.strada_cycle_thread.start()

    def stop_workflow(self):
        self.workflow_id += 1
        self.scan_active = False

        self.strada_cycle_running = False
        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.autolevel_done_for_current_tub = False
        self.scanning_in_progress = False
        self.drain_lost_timestamp = None

        self.leveller.stop()
        self.drainer.stop()
        self.drain_watcher.stop()

        # Halt all motors & window state on manual stop
        self.gui.force_drain_window = False
        try:
            self.gui.hw.move_up(False)
            self.gui.hw.move_down(False)
            self.gui.hw.move_forward(False)
            self.gui.hw.move_backward(False)
            self.gui.hw.rotate_cw(False)
            self.gui.hw.rotate_ccw(False)
        except Exception:
            pass