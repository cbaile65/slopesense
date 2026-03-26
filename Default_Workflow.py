import threading
import time


class StradaWorkflow:
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
        self.center_abort_requested = False
        self.last_drain_seen_state = False
        self.autolevel_done_for_current_tub = False

    def start_workflow(self):
        self.scan_active = True
        self.workflow_id += 1
        current_workflow_id = self.workflow_id

        self.strada_cycle_running = False
        self.current_tub_processed = False
        self.current_tub_has_drain = False
        self.center_abort_requested = False
        self.last_drain_seen_state = False
        self.autolevel_done_for_current_tub = False

        self.gui.on_drain_status_changed(False)
        if hasattr(self.gui.camera, "disable_defect_search"):
            self.gui.camera.disable_defect_search(clear_box=True)

        self.drain_watcher.start(status_callback=self.gui.on_drain_status_changed)

        def _workflow_manager():
            print("[Strada Workflow] Running: autolevel first, then wait for drain.")
            try:
                while self.scan_active and self.workflow_id == current_workflow_id:
                    drain_seen = bool(self.drain_watcher.drain_present)
                    drain_locked = bool(self.drain_watcher.locked_drain_ready)

                    if not drain_seen and self.last_drain_seen_state:
                        print("[Strada Workflow] Drain disappeared. Re-arming next cycle.")

                        if self.current_tub_processed:
                            print("[Strada Workflow] Tub successfully processed and removed. Incrementing count.")
                            self.gui.root.after(0, self.gui.increment_tub_count)

                        self.current_tub_has_drain = False
                        self.current_tub_processed = False
                        self.autolevel_done_for_current_tub = False

                        if hasattr(self.gui.camera, "disable_defect_search"):
                            self.gui.root.after(0, lambda: self.gui.camera.disable_defect_search(clear_box=True))

                        if self.strada_cycle_running and self.drainer.is_running:
                            print("[Strada Workflow] Drain lost during centering. Aborting cycle.")
                            self.center_abort_requested = True
                            self.drainer.stop()
                            self.gui.root.after(0, self.gui.close_drain_window)

                    if not self.autolevel_done_for_current_tub and not self.strada_cycle_running:
                        if hasattr(self.gui.camera, "disable_defect_search"):
                            self.gui.root.after(0, lambda: self.gui.camera.disable_defect_search(clear_box=True))
                        print("[Strada Workflow] Starting autolevel for next tub.")
                        self.start_strada_cycle(current_workflow_id, mode="autolevel_then_wait_for_drain")

                    elif self.autolevel_done_for_current_tub and not self.current_tub_processed and not self.strada_cycle_running:
                        fresh_lock = self.drain_watcher.consume_new_lock_event()

                        if fresh_lock or drain_locked:
                            print("[Strada Workflow] Drain ready. Starting centering cycle.")
                            self.current_tub_has_drain = True
                            self.start_strada_cycle(current_workflow_id, mode="center_only")

                    self.last_drain_seen_state = drain_seen
                    time.sleep(0.05)

            except Exception as e:
                print(f"[Workflow Manager Error] {e}")
                self.gui.root.after(0, self.gui.stop_all_workflows)

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

                if mode == "autolevel_then_wait_for_drain":
                    print("[Strada Cycle] Starting autolevel.")
                    self.gui.root.after(0, lambda: self.gui.btn_auto.config(bg="green"))
                    self.leveller.start()

                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.leveller.is_running:
                        time.sleep(0.05)

                    self.gui.root.after(0, lambda: self.gui.btn_auto.config(bg="#A9A9A9"))

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    self.autolevel_done_for_current_tub = True
                    print("[Strada Cycle] Autolevel complete. Waiting for drain.")

                elif mode == "center_only":
                    if not self.drain_watcher.drain_present:
                        print("[Strada Cycle] Drain not present anymore. Canceling centering.")
                        return

                    print("[Strada Cycle] Starting autocenter.")
                    self.gui.root.after(0, lambda: self.gui.btn_drain.config(bg="green"))
                    self.gui.root.after(0, self.gui.open_drain_window)

                    self.drainer.start(
                        callback=self.gui.auto_drain_done,
                        stop_on_first_drain=False,
                    )

                    while self.scan_active and self.workflow_id == workflow_id_at_start:
                        if not self.drain_watcher.drain_present:
                            print("[Strada Cycle] Drain lost during autocenter. Stopping centering.")
                            self.center_abort_requested = True
                            self.drainer.stop()
                            self.gui.root.after(0, self.gui.close_drain_window)
                            break

                        if not self.drainer.is_running:
                            print("[Strada Cycle] Drainer reported complete.")
                            break

                        time.sleep(0.05)

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    if self.center_abort_requested:
                        print("[Strada Cycle] Cycle aborted. Waiting for drain loss / next tub.")
                        return

                    print("[Strada Cycle] Autocenter complete. Stopping drainer before delay.")
                    self.drainer.stop()
                    time.sleep(0.1)

                    print("[Strada Cycle] Waiting 4.5 seconds before starting defect/edge detection.")
                    delay_start = time.time()

                    # --- DEBOUNCE LOGIC ---
                    drain_loss_time = None
                    debounce_duration = 4.0  # Increased to 4.0 seconds

                    # NOTE: Increased the total loop time to 4.5 seconds so a 4-second drop can actually be measured.
                    while self.scan_active and self.workflow_id == workflow_id_at_start and (
                            time.time() - delay_start) < 4.5:
                        if not self.drain_watcher.drain_present:
                            if drain_loss_time is None:
                                drain_loss_time = time.time()  # Start timing the loss
                            elif (time.time() - drain_loss_time) >= debounce_duration:
                                print(
                                    f"[Strada Cycle] Drain lost continuously for {debounce_duration}s during post-center delay. Canceling defect search.")
                                return
                        else:
                            drain_loss_time = None  # Reset the timer if the drain comes back

                        time.sleep(0.05)
                    # ----------------------

                    if not self.scan_active or self.workflow_id != workflow_id_at_start:
                        return

                    print("[Strada Cycle] Delay complete. Starting defect/edge detection.")
                    if hasattr(self.gui.camera, "enable_defect_search"):
                        self.gui.root.after(0, lambda: self.gui.camera.enable_defect_search(reset_timer=True))

                    self.current_tub_processed = True

            except Exception as e:
                print(f"[Strada Cycle Error] {e}")
                self.gui.root.after(0, self.gui.stop_all_workflows)

            finally:
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
        self.center_abort_requested = False
        self.last_drain_seen_state = False
        self.autolevel_done_for_current_tub = False

        if hasattr(self.gui.camera, "disable_defect_search"):
            self.gui.root.after(0, lambda: self.gui.camera.disable_defect_search(clear_box=True))

        self.leveller.stop()
        self.drainer.stop()
        self.drain_watcher.stop()