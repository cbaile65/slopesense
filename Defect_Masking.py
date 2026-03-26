import pyrealsense2 as rs
import numpy as np
import cv2
import time

# ----------------- Settings -----------------
W, H, FPS = 640, 480, 30
TIMEOUT_MS = 10000

# Keeps the depth map from "smearing" while the tub moves
DEPTH_STACK_N = 8

Z_MIN_M, Z_MAX_M = 0.15, 2.00
BAND_HALF_M = 0.02

# --- DIVOT SAVERS ---
SMOOTH_K = 3
MIN_BBOX_W_IN = 2
MIN_BBOX_H_IN = 2

FG_MIN_AREA = 3000
INTERIOR_ERODE_PX = 31

MIN_DEFECT_AREA_PX = 200
POST_CLOSE_K = 2

# --- DRAIN IGNORER ---
IGNORE_OUTER_ENDS_PCT = 0.20

# --- HYPER-AGGRESSIVE SPATIAL SENSITIVITY ---
AUTO_SIGMA_MULT = 1.2
CENTER_SENSITIVITY_MULT = 0.6
MIN_DEFECT_MM = 0.4
MAX_DEFECT_MM = 8.0
EDGE_PENALTY_MULT = 12.0

# NEW: Stretches the high-sensitivity zone vertically.
# 1.0 = perfect circle. 4.0 = wide, forgiving vertical pill shape.
CENTER_Y_FORGIVENESS = 4.0

LOCAL_TREND_SIGMA = 31
RESIDUAL_BLUR_K = 7

# --- TEMPORAL GATEKEEPER SETTINGS ---
PERSIST_N = 24
PERSIST_MIN_HITS = 14
PERSIST_DILATE_K = 7

# How many seconds the visible defect must persist before the box locks
BOX_LOCK_DELAY_SEC = 5.0

MM_PER_INCH = 25.4

SHOW_DIVOTS = True


# ----------------- Helpers -----------------
def largest_component(mask_u8, min_area=500):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8)
    best_i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    if stats[best_i, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(mask_u8)

    out = np.zeros_like(mask_u8)
    x, y, w, h = (
        stats[best_i, cv2.CC_STAT_LEFT],
        stats[best_i, cv2.CC_STAT_TOP],
        stats[best_i, cv2.CC_STAT_WIDTH],
        stats[best_i, cv2.CC_STAT_HEIGHT],
    )
    out[y:y + h, x:x + w][labels[y:y + h, x:x + w] == best_i] = 255
    return out


def filter_components(mask_u8, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            out[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i] = 255
    return out


def robust_sigma(vals):
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return 0.0
    return 1.4826 * np.median(np.abs(vals - np.median(vals)))


def make_interior_mask(fg_mask_u8, erode_px):
    if erode_px < 3:
        return fg_mask_u8.copy()
    return cv2.erode(
        fg_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px)),
        iterations=1
    )


def draw_mask_outline(img, mask_u8, color, thickness=2):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


def fit_quadratic_surface(depth, mask):
    ys, xs = np.where(mask)
    if len(xs) < 1200:
        return (
            np.array([0, 0, 0, 0, 0, float(np.median(depth[mask]))], dtype=np.float32)
            if np.any(mask)
            else np.zeros(6, dtype=np.float32)
        )
    x, y, z = xs.astype(np.float32), ys.astype(np.float32), depth[ys, xs].astype(np.float32)
    coeffs, *_ = np.linalg.lstsq(
        np.stack([x * x, y * y, x * y, x, y, np.ones_like(x)], axis=1),
        z,
        rcond=None
    )
    return coeffs.astype(np.float32)


def quadratic_surface_img(coeffs, h, w):
    a, b, c, d, e, f = coeffs
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return a * X * X + b * Y * Y + c * X * Y + d * X + e * Y + f


def masked_local_trend(img, mask, sigma):
    mask_f = mask.astype(np.float32)
    num = cv2.GaussianBlur((img * mask_f).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return num / (den + 1e-6)


def filter_components_physical(mask_u8, depth_s, fx, fy, min_w_in, min_h_in):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    min_w_mm, min_h_mm = min_w_in * MM_PER_INCH, min_h_in * MM_PER_INCH

    for i in range(1, num):
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        if stats[i, cv2.CC_STAT_AREA] <= 0:
            continue

        comp_mask = (labels[y:y + h, x:x + w] == i)
        z_vals = depth_s[y:y + h, x:x + w][comp_mask]
        z_vals = z_vals[(z_vals > 0) & np.isfinite(z_vals)]
        if z_vals.size == 0:
            continue

        z_med_mm = float(np.median(z_vals)) * 1000.0
        if (w * z_med_mm / fx) >= min_w_mm and (h * z_med_mm / fy) >= min_h_mm:
            out[y:y + h, x:x + w][comp_mask] = 255
    return out


# ----------------- Main Class -----------------
class DefectDetector:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

        self.profile = self.pipeline.start(config)
        self.device = self.profile.get_device()
        self.depth_scale = self.device.first_depth_sensor().get_depth_scale()

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = float(intr.fx), float(intr.fy)

        self.align = rs.align(rs.stream.color)

        self.k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k3 = np.ones((3, 3), np.uint8)
        self.kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (POST_CLOSE_K, POST_CLOSE_K))
        self.kvote = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PERSIST_DILATE_K, PERSIST_DILATE_K))

        self.depth_stack = np.zeros((DEPTH_STACK_N, H, W), dtype=np.float32)
        self.depth_count = 0
        self.div_history = np.zeros((PERSIST_N, H, W), dtype=np.uint8)
        self.div_count = 0

        self.scan_start_time = time.time()
        self.locked_relative_box = None
        self.defect_search_enabled = False
        self.defect_first_seen_time = None

        # Pre-compute coordinate grids for independent X/Y sensitivity mapping
        self.X_grid, self.Y_grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames(TIMEOUT_MS)
        print("Warmup complete.")

        self.ref_depth = None

    def enable_defect_search(self, reset_timer=True):
        self.defect_search_enabled = True
        self.div_count = 0
        self.div_history[:] = 0
        self.defect_first_seen_time = None

        # --- CRITICAL FIX ---
        # Flush the old reference depth and old stack frames
        # so it learns the new tub height immediately after centering.
        self.ref_depth = None
        self.depth_count = 0
        # --------------------

        if reset_timer:
            self.scan_start_time = time.time()
            self.locked_relative_box = None

    def disable_defect_search(self, clear_box=True):
        self.defect_search_enabled = False
        self.div_count = 0
        self.div_history[:] = 0
        self.defect_first_seen_time = None
        if clear_box:
            self.locked_relative_box = None
        self.scan_start_time = time.time()

    def reset_depth(self):
        self.ref_depth = None
        self.depth_count = 0
        self.div_count = 0
        self.div_history[:] = 0
        self.scan_start_time = time.time()
        self.locked_relative_box = None
        self.defect_search_enabled = False
        self.defect_first_seen_time = None

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except RuntimeError:
            return None

        frames = self.align.process(frames)
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if not cf or not df:
            return None

        color = np.asanyarray(cf.get_data()).copy()
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * self.depth_scale

        del cf, df, frames

        self.depth_stack[self.depth_count % DEPTH_STACK_N] = depth_m
        self.depth_count += 1

        # If we just reset the stack, return clean color until we have a stable median
        if self.depth_count < DEPTH_STACK_N:
            return color.copy()

        depth_med = np.median(self.depth_stack, axis=0).astype(np.float32)

        depth_masking = cv2.bilateralFilter(depth_med, d=9, sigmaColor=0.05, sigmaSpace=15)
        depth_defects = cv2.GaussianBlur(depth_med, (SMOOTH_K, SMOOTH_K), 0) if SMOOTH_K >= 3 else depth_med

        if self.ref_depth is None:
            cy, cx = H // 2, W // 2
            half_box = 20
            center_depths = depth_med[cy - half_box: cy + half_box, cx - half_box: cx + half_box]
            valid_depths = center_depths[(center_depths > Z_MIN_M) & (center_depths < Z_MAX_M)]
            if valid_depths.size > 0:
                self.ref_depth = float(np.median(valid_depths))

        fg_u8 = np.zeros((H, W), dtype=np.uint8)
        outer_tub_u8 = np.zeros((H, W), dtype=np.uint8)

        if self.ref_depth is not None:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            _, rgb_mask = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            band = (depth_masking > Z_MIN_M) & (depth_masking < Z_MAX_M) & (
                    np.abs(depth_masking - self.ref_depth) <= BAND_HALF_M
            )
            raw_inner_depth = np.zeros((H, W), dtype=np.uint8)
            raw_inner_depth[band] = 255

            raw_inner_depth = cv2.morphologyEx(
                cv2.morphologyEx(raw_inner_depth, cv2.MORPH_OPEN, self.k_large, iterations=1),
                cv2.MORPH_CLOSE,
                self.k_large,
                iterations=2
            )
            raw_inner_depth = largest_component(raw_inner_depth, FG_MIN_AREA)

            search_area_inner = cv2.dilate(raw_inner_depth, self.k_large, iterations=2)

            fg_u8_temp = cv2.bitwise_and(rgb_mask, search_area_inner)
            fg_u8_temp = cv2.morphologyEx(fg_u8_temp, cv2.MORPH_CLOSE, self.k_large, iterations=1)
            fg_u8_temp = largest_component(fg_u8_temp, FG_MIN_AREA)

            rect_inner = None
            contours_inner, _ = cv2.findContours(fg_u8_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_inner:
                c_inner = max(contours_inner, key=cv2.contourArea)
                rect_inner = cv2.minAreaRect(c_inner)
                box_inner = cv2.boxPoints(rect_inner)
                box_inner = np.intp(box_inner)
                cv2.drawContours(fg_u8, [box_inner], 0, 255, thickness=cv2.FILLED)

            if rect_inner is not None:
                outer_search_area = cv2.dilate(raw_inner_depth, self.k_large, iterations=7)

                hybrid_outer = cv2.bitwise_and(rgb_mask, outer_search_area)
                hybrid_outer = cv2.morphologyEx(hybrid_outer, cv2.MORPH_OPEN, self.k_large, iterations=1)
                hybrid_outer = cv2.morphologyEx(hybrid_outer, cv2.MORPH_CLOSE, self.k_large, iterations=2)
                hybrid_outer = largest_component(hybrid_outer, FG_MIN_AREA)

                contours_outer, _ = cv2.findContours(hybrid_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_outer:
                    c_outer = max(contours_outer, key=cv2.contourArea)

                    (cx, cy), (w, h), angle = rect_inner
                    C = np.array([cx, cy])

                    box_pts = cv2.boxPoints(rect_inner)
                    u1 = box_pts[1] - box_pts[0]
                    u2 = box_pts[2] - box_pts[1]

                    L1, L2 = np.linalg.norm(u1), np.linalg.norm(u2)

                    if L1 > 1e-3 and L2 > 1e-3:
                        u1_hat, u2_hat = u1 / L1, u2 / L2

                        pts = c_outer.reshape(-1, 2) - C
                        proj1 = np.abs(pts.dot(u1_hat))
                        proj2 = np.abs(pts.dot(u2_hat))

                        max_p1 = np.percentile(proj1, 99.5)
                        max_p2 = np.percentile(proj2, 99.5)

                        pad1 = max_p1 - (L1 / 2.0)
                        pad2 = max_p2 - (L2 / 2.0)

                        padding = pad1 if L1 < L2 else pad2
                        padding = np.clip(padding, 5, 80)

                        new_w = w + (2 * padding)
                        new_h = h + (2 * padding)

                        rect_outer = ((cx, cy), (new_w, new_h), angle)
                        box_outer = np.intp(cv2.boxPoints(rect_outer))
                        cv2.drawContours(outer_tub_u8, [box_outer], 0, 255, thickness=cv2.FILLED)

        fg_mask = fg_u8 > 0
        highlight = color.copy()

        if not np.any(fg_mask):
            if self.locked_relative_box is not None:
                self.locked_relative_box = None
            self.defect_first_seen_time = None
            if not self.defect_search_enabled:
                self.div_count = 0
                self.div_history[:] = 0
            return highlight

        interior_u8 = make_interior_mask(fg_u8, INTERIOR_ERODE_PX)
        interior_mask = interior_u8 > 0
        if np.count_nonzero(interior_mask) < 1500:
            interior_u8, interior_mask = fg_u8.copy(), fg_mask

        if np.any(interior_u8):
            bx, by, bw, bh = cv2.boundingRect(interior_u8)
            exclude_left_x = bx + int(bw * IGNORE_OUTER_ENDS_PCT)
            exclude_right_x = bx + int(bw * (1.0 - IGNORE_OUTER_ENDS_PCT))

            interior_mask[:, :exclude_left_x] = False
            interior_mask[:, exclude_right_x:] = False

        if self.defect_search_enabled:
            dist_map = cv2.distanceTransform(interior_u8, cv2.DIST_L2, 5)
            max_dist = dist_map.max() + 1e-6
            dist_norm = dist_map / max_dist

            quad = quadratic_surface_img(fit_quadratic_surface(depth_defects, interior_mask), H, W)
            resid_local_s = cv2.GaussianBlur(
                (depth_defects - quad) - masked_local_trend(depth_defects - quad, interior_mask, LOCAL_TREND_SIGMA),
                (RESIDUAL_BLUR_K, RESIDUAL_BLUR_K),
                0
            )

            sigma = robust_sigma(resid_local_s[interior_mask])
            base_thresh = np.clip(AUTO_SIGMA_MULT * sigma, MIN_DEFECT_MM / 1000.0, MAX_DEFECT_MM / 1000.0)

            # Preserve the physical edge penalty (keeps normal sensitivity at tub walls)
            edge_proximity = 1.0 - dist_norm
            thresh_map = base_thresh * (1.0 + (edge_proximity ** 3) * (EDGE_PENALTY_MULT - 1.0))

            # --- NEW: Independent X/Y Math for the "Vertical Pill" sweet spot ---
            bx, by, bw, bh = cv2.boundingRect(interior_u8)
            cx, cy = bx + bw / 2.0, by + bh / 2.0

            # Calculate linear distance from center (0 = at center, 1 = at tub edge)
            norm_x = np.clip(np.abs(self.X_grid - cx) / (bw / 2.0 + 1e-6), 0, 1)
            norm_y = np.clip(np.abs(self.Y_grid - cy) / (bh / 2.0 + 1e-6), 0, 1)

            # Apply the power curve strictly to the Y axis to flatten the slope and make it forgiving up/down
            norm_y_forgiving = norm_y ** CENTER_Y_FORGIVENESS

            # Recombine into a custom distance map and invert it for the proximity penalty
            custom_norm_dist = np.clip(np.sqrt(norm_x ** 2 + norm_y_forgiving ** 2), 0, 1)
            center_proximity = 1.0 - custom_norm_dist

            # Apply the highly-forgiving center sensitivity drop
            thresh_map = thresh_map * (1.0 - center_proximity * (1.0 - CENTER_SENSITIVITY_MULT))

            raw_defect_mask = (interior_mask & (resid_local_s >= thresh_map)).astype(np.uint8) * 255

            div_u8 = filter_components(
                cv2.morphologyEx(
                    cv2.morphologyEx(raw_defect_mask, cv2.MORPH_OPEN, self.k5, iterations=1),
                    cv2.MORPH_CLOSE,
                    self.kclose,
                    iterations=1
                ),
                MIN_DEFECT_AREA_PX
            )

            self.div_history[self.div_count % PERSIST_N] = (
                    cv2.dilate(div_u8, self.kvote, iterations=1) > 0
            ).astype(np.uint8)
            self.div_count += 1

            if self.div_count > 0:
                history_gating_mask = (
                        np.sum(self.div_history[:min(self.div_count, PERSIST_N)], axis=0) >= PERSIST_MIN_HITS
                )
                div_mask = history_gating_mask & (div_u8 > 0)
            else:
                div_mask = div_u8 > 0

            div_final_u8 = filter_components_physical(
                filter_components((div_mask.astype(np.uint8) * 255), MIN_DEFECT_AREA_PX),
                depth_defects,
                self.fx,
                self.fy,
                MIN_BBOX_W_IN,
                MIN_BBOX_H_IN
            )

            overlay = highlight.copy()
            smoothed_mask = np.zeros_like(div_final_u8)

            if SHOW_DIVOTS and np.any(div_final_u8):
                blurred_div = cv2.GaussianBlur(div_final_u8, (21, 21), 0)
                _, smoothed_mask = cv2.threshold(blurred_div, 127, 255, cv2.THRESH_BINARY)

                if self.locked_relative_box is None:
                    overlay[smoothed_mask > 0] = (0, 0, 255)

            highlight = cv2.addWeighted(overlay, 0.45, highlight, 0.55, 0)

            defect_visible_now = np.any(smoothed_mask)

            if self.locked_relative_box is None:
                if defect_visible_now:
                    if self.defect_first_seen_time is None:
                        self.defect_first_seen_time = time.time()
                else:
                    self.defect_first_seen_time = None

            if self.locked_relative_box is None and defect_visible_now and self.defect_first_seen_time is not None:
                if (time.time() - self.defect_first_seen_time) >= BOX_LOCK_DELAY_SEC:
                    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours and np.any(interior_u8):
                        c = max(contours, key=cv2.contourArea)
                        dx, dy, dw, dh = cv2.boundingRect(c)

                        tx, ty, tw, th = cv2.boundingRect(interior_u8)

                        rel_x = (dx - tx) / float(tw)
                        rel_y = (dy - ty) / float(th)
                        rel_w = dw / float(tw)
                        rel_h = dh / float(th)

                        self.locked_relative_box = (rel_x, rel_y, rel_w, rel_h)

            if self.locked_relative_box is not None and np.any(interior_u8):
                rel_x, rel_y, rel_w, rel_h = self.locked_relative_box
                tx, ty, tw, th = cv2.boundingRect(interior_u8)

                dx = int(tx + rel_x * tw)
                dy = int(ty + rel_y * th)
                dw = int(rel_w * tw)
                dh = int(rel_h * th)

                pad = 15
                x1, y1 = max(0, dx - pad), max(0, dy - pad)
                x2, y2 = min(W, dx + dw + pad), min(H, dy + dh + pad)

                cv2.rectangle(highlight, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    highlight,
                    "DEFECT",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            if np.any(outer_tub_u8):
                draw_mask_outline(highlight, outer_tub_u8, (0, 255, 255), 1)
            else:
                draw_mask_outline(highlight, fg_u8, (0, 255, 255), 1)

            draw_mask_outline(highlight, interior_u8, (0, 255, 0), 2)

        else:
            self.div_count = 0
            self.div_history[:] = 0
            self.defect_first_seen_time = None

        return highlight

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass