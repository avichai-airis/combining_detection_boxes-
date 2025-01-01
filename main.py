from collections import defaultdict

import numpy as np
import cv2
from cython_bbox import bbox_overlaps as bbox_ious
import lap
import json
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from matching_analysis_plots import (
    plot_size_vs_confidence_per_class,
    plot_matches_per_class,
    plot_confidence_histogram,
    plot_confidence_histogram_per_class,
    plot_box_size_histogram_per_class,
    plot_confidence_histograms_multiple_iou,
)

from plot_utils import draw_bbox, show_image


class BoundingBox:

    def __init__(self, xtl, ytl, xbr, ybr, class_name=None, confidence=None, embedding=None):
        self.xtl = max(xtl, 0)
        self.ytl = max(ytl, 0)
        self.xbr = max(xbr, 0)
        self.ybr = max(ybr, 0)
        self.class_name = class_name
        self.confidence = confidence
        self.embedding = embedding
        self.is_matched = False
        self.matched_bbox = None

    def __str__(self):
        return (
            f"BoundingBox({self.xtl=}, {self.ytl=}, {self.xbr=}, {self.ybr=}, {self.class_name=}, {self.confidence=})"
        )

    def __repr__(self):
        return self.__str__()

    def set_matched_bbox(self, matched_bbox):
        self.is_matched = True
        self.matched_bbox = matched_bbox

    def get_tlbr(self):
        """
        Get the top-left and bottom-right coordinates of the bounding box - [xtl, ytl, xbr, ybr]
        """
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def draw(self, image, show_label=True, color=(0, 255, 0), thickness=2):
        """Draw this bounding box on the image"""
        draw_bbox(image, self, show_label, color, thickness)


class FrameDetections:
    def __init__(self, timestamp=None, image_size=None):
        self.timestamp = timestamp
        self.image_size = image_size or []
        self.detections = []

    @classmethod
    def from_dict(cls, frame_data: dict):
        """Create FrameDetections from dictionary format"""
        frame = cls(frame_data.get("timestamp"), frame_data.get("imagesize", []))
        for detection in frame_data.get("objects", []):
            xtl, ytl, xbr, ybr = map(np.float64, detection["box"])
            class_name = detection["class"]
            confidence = detection["confidence"]
            frame.detections.append(BoundingBox(xtl, ytl, xbr, ybr, class_name, confidence))
        return frame

    @classmethod
    def from_boxes_and_embeddings(cls, boxes: np.ndarray, embeddings: np.ndarray):
        """Create FrameDetections from raw boxes array and optional embeddings"""
        frame = cls()
        for box, embedding in zip(boxes, embeddings):
            xtl, ytl, xbr, ybr = map(np.float64, box)
            frame.detections.append(BoundingBox(xtl, ytl, xbr, ybr, embedding=embedding))
        return frame

    def draw_detections(self, image, show_labels=True):
        """Draw all detections on the image"""
        img_copy = image.copy()
        for detection in self.detections:
            detection.draw(img_copy, show_labels)
        return img_copy

    def show_detections(self, image, show_labels=True):
        """Display image with detections"""
        img_with_detections = self.draw_detections(image, show_labels)
        show_image(img_with_detections, "Detections")


class DetectionsManager:
    def __init__(self, data):
        if isinstance(data, str) and data.endswith(".json"):
            with open(data, "r") as f:
                json_data = json.load(f)
            self.frames = {
                fn: FrameDetections.from_dict(data) for fn, data in json_data.get("obj_detection", {}).items()
            }
        elif isinstance(data, dict):
            self.frames = {fn: FrameDetections.from_dict(data) for fn, data in data.items()}
        else:
            raise ValueError("Invalid input format. Must be either JSON path or preprocessed data dictionary")

    @classmethod
    def from_npz(cls, npz_path: str):
        """Create DetectionsManager instance from NPZ file"""
        npz_data = np.load(npz_path)
        boxes = npz_data["boxes"]
        embeddings = npz_data["embeddings"]
        frame_names = npz_data["frame_names"]
        frames_set = set(frame_names)
        frames = defaultdict(list)
        for frame_name in frames_set:
            frame_idx = np.where(frame_names == frame_name)[0]
            frames[frame_name] = (boxes[frame_idx], embeddings[frame_idx])

        # Create instance with empty dictionary
        instance = cls.__new__(cls)
        instance.frames = {
            frame_name: FrameDetections.from_boxes_and_embeddings(frame_boxes, frame_embeddings)
            for frame_name, (frame_boxes, frame_embeddings) in frames.items()
        }
        return instance

    def get_frame_detections(self, frame_name: str) -> FrameDetections or None:
        return self.frames.get(frame_name)

    def get_all_bbox_in_frame(self, frame_name: str) -> list[list]:
        frame = self.get_frame_detections(frame_name)
        if frame:
            return [detection.get_tlbr() for detection in frame.detections]
        return []

    def list_frames(self) -> list:
        return list(self.frames.keys())

    def all_detections(self):
        for fn, frame in self.frames.items():
            yield fn, frame.detections

    def get_all_frame_names(self):
        for fn in self.frames.keys():
            yield fn

    def show_frame_detections(self, frame_name: str, image, show_labels=True):
        """Show detections for a specific frame"""
        frame = self.get_frame_detections(frame_name)
        if frame:
            frame.show_detections(image, show_labels)
        else:
            print(f"No detections found for frame: {frame_name}")


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


class BBoxMatcher:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold

    def match_bboxes(self, bboxes1, bboxes2):
        if not bboxes1 or not bboxes2:
            return [], list(range(len(bboxes1))), list(range(len(bboxes2)))

        bboxes1_array = np.array(bboxes1)
        bboxes2_array = np.array(bboxes2)

        cost_matrix = self._iou_distance(bboxes1_array, bboxes2_array)
        matches, unmatched_bboxes1, unmatched_bboxes2 = self._linear_assignment(cost_matrix, 1 - self.iou_threshold)

        return matches, unmatched_bboxes1, unmatched_bboxes2

    def _ious(self, atlbrs, btlbrs):
        """
        Compute cost based on IoU
        :type atlbrs: list[tlbr] | np.ndarray
        :type atlbrs: list[tlbr] | np.ndarray

        :rtype ious np.ndarray
        """
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
        if ious.size == 0:
            return ious

        ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float64), np.ascontiguousarray(btlbrs, dtype=np.float64))

        return ious

    def _linear_assignment(self, cost_matrix, thresh):
        # empty cost matrix
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []

        # LAPJV algorithm
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

        # matches
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])

        # unmatched
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]

        return np.asarray(matches), unmatched_a, unmatched_b

    def _iou_distance(self, atlbrs, btlbrs):
        """
        Compute cost based on IoU for a list of bounding boxes.
        """
        return 1 - self._ious(atlbrs, btlbrs)


def get_video_names(closed_path: str, open_path: str) -> list:
    """Get list of videos that exist in both directories."""
    closed_videos = [f for f in os.listdir(closed_path) if f.endswith(".mp4")]
    open_videos = [f.split(".npz")[0] for f in os.listdir(open_path) if f.endswith(".mp4.npz")]
    return list(set(closed_videos).intersection(set(open_videos)))


def print_missing_videos(closed_videos: list, open_videos: list, common_videos: list) -> None:
    """Print videos that are not present in both folders."""
    for video in closed_videos:
        if video not in common_videos:
            print(f"video {video} is not in open folder")
    for video in open_videos:
        if video not in common_videos:
            print(f"video {video} is not in closed folder")


def get_detection_paths(video_name: str, closed_path: str, open_path: str) -> tuple:
    """Generate paths for detection files."""
    detection_path = os.path.join(closed_path, video_name, "object_detection.json")
    npz_path = os.path.join(open_path, f"{video_name}.npz")
    return detection_path, npz_path


def process_frame_detections(json_detections, npz_detections, bbox_matcher, frame_name):
    """Process and match detections for a single frame."""
    # Create empty FrameDetections if frame exists in NPZ but not in JSON
    if frame_name not in json_detections.frames:
        json_detections.frames[frame_name] = FrameDetections()
    if frame_name not in npz_detections.frames:
        npz_detections.frames[frame_name] = FrameDetections()

    j_bboxes = json_detections.get_all_bbox_in_frame(frame_name)
    n_bboxes = npz_detections.get_all_bbox_in_frame(frame_name)
    matches, _, _ = bbox_matcher.match_bboxes(j_bboxes, n_bboxes)

    for idx1, idx2 in matches:
        json_detections.frames[frame_name].detections[idx1].set_matched_bbox(
            npz_detections.frames[frame_name].detections[idx2]
        )
        npz_detections.frames[frame_name].detections[idx2].set_matched_bbox(
            json_detections.frames[frame_name].detections[idx1]
        )


def match_multi_source_detections(iou_threshold=0.6):
    closed_path = "/Users/avichai/Documents/projects/combining_detection_boxes/data/closed/"
    open_path = "/Users/avichai/Documents/projects/combining_detection_boxes/data/open"
    bbox_matcher = BBoxMatcher(iou_threshold)

    videos_name = get_video_names(closed_path, open_path)
    print_missing_videos(
        [f for f in os.listdir(closed_path) if f.endswith(".mp4")],
        [f.split(".npz")[0] for f in os.listdir(open_path) if f.endswith(".mp4.npz")],
        videos_name,
    )

    jsons_detections_manager = {}
    npz_detections_manager = {}

    for video_name in tqdm(videos_name, desc="videos"):
        detection_path, npz_path = get_detection_paths(video_name, closed_path, open_path)

        json_detections = DetectionsManager(detection_path)
        npz_detections = DetectionsManager.from_npz(npz_path)

        # Get all unique frame names from both sources
        frame_names = set(list(json_detections.get_all_frame_names()) + list(npz_detections.get_all_frame_names()))

        for frame_name in frame_names:
            process_frame_detections(json_detections, npz_detections, bbox_matcher, frame_name)

        jsons_detections_manager[video_name] = json_detections
        npz_detections_manager[video_name] = npz_detections

    return jsons_detections_manager, npz_detections_manager


def visualize_detections_for_video(video_name):
    closed_path = "/Users/avichai/Documents/projects/combining_detection_boxes/data/closed/"
    open_path = "/Users/avichai/Documents/projects/combining_detection_boxes/data/open"
    bbox_matcher = BBoxMatcher(0.6)

    detection_path, npz_path = get_detection_paths(video_name, closed_path, open_path)
    json_detections = DetectionsManager(detection_path)
    npz_detections = DetectionsManager.from_npz(npz_path)
    # Get all unique frame names from both sources
    frame_names = set(list(json_detections.get_all_frame_names()) + list(npz_detections.get_all_frame_names()))

    for frame_name in frame_names:
        process_frame_detections(json_detections, npz_detections, bbox_matcher, frame_name)

    frames_path = (
        f"/Users/avichai/Documents/projects/combining_detection_boxes/data/videos_sampling/{video_name}/sampled_images"
    )
    visualizer = InteractiveVisualizer(frames_path, json_detections, npz_detections)
    visualizer.run()


if __name__ == "__main__":

    # plot_confidence_histograms_multiple_iou(match_multi_source_detections, save_fig=False)
    save_fig = False
    jsons_detections_manager, npz_detections_manager = match_multi_source_detections()
    plot_size_vs_confidence_per_class(jsons_detections_manager, save_fig)
    plot_matches_per_class(jsons_detections_manager, npz_detections_manager, save_fig)
    plot_confidence_histogram(jsons_detections_manager, save_fig)
    plot_confidence_histogram_per_class(jsons_detections_manager, save_fig)
    plot_box_size_histogram_per_class(jsons_detections_manager, save_fig)
