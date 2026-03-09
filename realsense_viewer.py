import pyrealsense2 as rs
import numpy as np
import cv2

# Camera settings
W, H, FPS = 640, 480, 30

# Cursor state
cursor = [W//2, H//2, 0]  # x, y, side (0 = RGB, 1 = depth)

def mouse(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:

        side = 0

        # If mouse is on the depth image (right half)
        if x >= W:
            x -= W
            side = 1

        # Clamp values to prevent out-of-bounds
        x = int(np.clip(x, 0, W-1))
        y = int(np.clip(y, 0, H-1))

        cursor[0] = x
        cursor[1] = y
        cursor[2] = side


# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

profile = pipeline.start(config)

# Depth scale (convert depth units -> meters)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

align = rs.align(rs.stream.color)

cv2.namedWindow("view")
cv2.setMouseCallback("view", mouse)

print("Depth scale:", depth_scale)

try:

    while True:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = depth * depth_scale

        # Depth visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET
        )

        x, y, side = cursor

        # Safe depth lookup
        d = depth_m[y, x]

        # Draw cursor
        cv2.circle(color, (x, y), 6, (0,255,255), -1)
        cv2.circle(depth_colormap, (x, y), 6, (0,255,255), -1)

        # Display depth value
        cv2.putText(
            color,
            f"Depth: {d:.3f} m",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        # Combine images
        combined = np.hstack((color, depth_colormap))

        cv2.imshow("view", combined)

        key = cv2.waitKey(1)

        if key == 27:
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()