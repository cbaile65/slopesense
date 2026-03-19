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

        # Always run passive drain watcher while START is active
        self.drain_watcher.start(status_callback=self.gui.on_drain_status_changed)

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
                            self.gui.root.after(0, self.gui.close_drain_window)

                    # Run autolevel first for each new tub before drain search matters
                    if not self.autolevel_done_for_current_tub and not self.strada_cycle_running:
                        print("[Strada Workflow] Starting autolevel for next tub.")
                        self.start_strada_cycle(current_workflow_id, mode="autolevel_then_wait_for_drain")

                    # After autolevel, allow either a fresh new lock event OR an already-held locked drain
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

                # ------------------------------------------
                # MODE 1: autolevel first, then return to watcher
                # ------------------------------------------
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

                # ------------------------------------------
                # MODE 2: drain already found, do center only
                # ------------------------------------------
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

                    while self.scan_active and self.workflow_id == workflow_id_at_start and self.drainer.is_running:
                        if not self.drain_watcher.drain_present:
                            print("[Strada Cycle] Drain lost during autocenter. Stopping centering.")
                            self.center_abort_requested = True
                            self.drainer.stop()
                            self.gui.root.after(0, self.gui.close_drain_window)
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

        self.leveller.stop()
        self.drainer.stop()
        self.drain_watcher.stop()