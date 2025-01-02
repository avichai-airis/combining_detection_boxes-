import cv2
import os


def draw_bbox(image, bbox, show_label=True, color=(0, 255, 0), thickness=2):
    """Draw a single bounding box on the image"""
    x1, y1, x2, y2 = map(int, [bbox.xtl, bbox.ytl, bbox.xbr, bbox.ybr])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if show_label:
        label = f"{bbox.class_name}: {bbox.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw label background
        cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        # Draw text
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)


def show_image(image, window_name="Image"):
    """Display image with OpenCV"""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class InteractiveVisualizer:
    def __init__(self, frames_path, json_detections, npz_detections):
        self.frames_path = frames_path
        self.frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith((".jpg", ".png"))])
        self.current_frame_idx = 0
        self.json_detections = json_detections
        self.npz_detections = npz_detections
        self.window_name = "Interactive Detections"

        # Colors for different types of detections
        self.json_unmatched_color = (0, 255, 0)  # Green for unmatched JSON
        self.npz_unmatched_color = (0, 0, 255)  # Red for unmatched NPZ
        self.json_matched_color = (255, 0, 0)  # Blue for matched JSON
        self.npz_matched_color = (0, 255, 255)  # Yellow for matched NPZ

        # Display modes
        self.show_all = True
        self.show_matched_only = False
        self.show_unmatched_json_only = False
        self.show_unmatched_npz_only = False

        # Load first frame
        self.load_current_frame()

    def load_current_frame(self):
        frame_name = self.frame_files[self.current_frame_idx]
        self.image = cv2.imread(f"{self.frames_path}/{frame_name}")
        self.original_image = self.image.copy()

        # Update window title with frame info
        cv2.setWindowTitle(
            self.window_name, f"Frame {self.current_frame_idx + 1}/{len(self.frame_files)} - {frame_name}"
        )

    def create_window(self):
        cv2.namedWindow(self.window_name)

    def update_image(self):
        self.image = self.original_image.copy()
        frame_name = self.frame_files[self.current_frame_idx]

        frame_dets_json = self.json_detections.get_frame_detections(frame_name)
        frame_dets_npz = self.npz_detections.get_frame_detections(frame_name)

        if frame_dets_json and frame_dets_npz:
            # Track match numbers
            match_number = 0
            matched_pairs = {}

            # First pass: build matched pairs dictionary and draw unmatched boxes
            for idx, det_json in enumerate(frame_dets_json.detections):
                if det_json.is_matched:
                    # Find the corresponding npz detection
                    for npz_idx, det_npz in enumerate(frame_dets_npz.detections):
                        if det_npz.matched_bbox == det_json:
                            matched_pairs[idx] = (npz_idx, match_number)
                            match_number += 1
                            break

            # Draw based on display mode
            for idx, det_json in enumerate(frame_dets_json.detections):
                if self.show_all or (
                    (self.show_matched_only and det_json.is_matched)
                    or (self.show_unmatched_json_only and not det_json.is_matched)
                ):
                    color = self.json_matched_color if det_json.is_matched else self.json_unmatched_color
                    det_json.draw(self.image, show_label=True, color=color)

                    # Add match number for matched boxes
                    if det_json.is_matched and idx in matched_pairs:
                        match_num = matched_pairs[idx][1]
                        cv2.putText(
                            self.image,
                            f"#{match_num}",
                            (int(det_json.xtl), int(det_json.ytl) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )

            for idx, det_npz in enumerate(frame_dets_npz.detections):
                if self.show_all or (
                    (self.show_matched_only and det_npz.is_matched)
                    or (self.show_unmatched_npz_only and not det_npz.is_matched)
                ):
                    color = self.npz_matched_color if det_npz.is_matched else self.npz_unmatched_color
                    det_npz.draw(self.image, show_label=False, color=color)

                    # Add match number for matched boxes
                    for json_idx, (npz_idx, match_num) in matched_pairs.items():
                        if npz_idx == idx:
                            cv2.putText(
                                self.image,
                                f"#{match_num}",
                                (int(det_npz.xtl), int(det_npz.ytl) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                            )

        cv2.imshow(self.window_name, self.image)

    def run(self):
        self.create_window()
        print("Options:")
        print("  p - Previous frame")
        print("  n - Next frame")
        print("  a - Show all detections")
        print("  m - Show only matched detections")
        print("  j - Show only unmatched JSON detections")
        print("  z - Show only unmatched NPZ detections")
        print("  q - Quit")

        while True:
            self.update_image()
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):  # Quit
                break
            elif key == ord("a"):  # Show all
                self.show_all = True
                self.show_matched_only = False
                self.show_unmatched_json_only = False
                self.show_unmatched_npz_only = False
            elif key == ord("m"):  # Show matched only
                self.show_all = False
                self.show_matched_only = True
                self.show_unmatched_json_only = False
                self.show_unmatched_npz_only = False
            elif key == ord("j"):  # Show unmatched JSON only
                self.show_all = False
                self.show_matched_only = False
                self.show_unmatched_json_only = True
                self.show_unmatched_npz_only = False
            elif key == ord("z"):  # Show unmatched NPZ only
                self.show_all = False
                self.show_matched_only = False
                self.show_unmatched_json_only = False
                self.show_unmatched_npz_only = True
            elif key == ord("n"):  # Next frame
                self.current_frame_idx = min(self.current_frame_idx + 1, len(self.frame_files) - 1)
                self.load_current_frame()
            elif key == ord("p"):  # Previous frame
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
                self.load_current_frame()

        cv2.destroyAllWindows()
