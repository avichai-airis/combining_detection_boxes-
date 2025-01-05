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

from plot_utils import draw_bbox, show_image, InteractiveVisualizer


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
    def __init__(self, npz_path: str):
        # Now accept only npz
        if not npz_path.endswith(".npz"):
            raise ValueError("DetectionsManager only supports .npz files.")
        self.frames = DetectionsManager.from_npz(npz_path).frames

    @classmethod
    def from_npz(cls, npz_path: str):
        npz_data = np.load(npz_path, allow_pickle=True)
        instance = cls.__new__(cls)
        frames_dict = defaultdict(list)

        # Distinguish between open (boxes, embeddings) and closed (bounding_boxes, confidences)
        if "embeddings" in npz_data:
            # "open" format
            frame_names = npz_data["frame_names"]
            boxes = npz_data["boxes"]
            embeddings = npz_data["embeddings"]
            for i, fn in enumerate(frame_names):
                frames_dict[fn].append((boxes[i], embeddings[i]))
            instance.frames = {}
            for fn, data_list in frames_dict.items():
                box_array = []
                emb_array = []
                for b, e in data_list:
                    box_array.append(b)
                    emb_array.append(e)
                instance.frames[fn] = FrameDetections.from_boxes_and_embeddings(
                    np.array(box_array), np.array(emb_array)
                )
        else:
            # "closed" format
            frame_names = npz_data["frame_names"]
            bounding_boxes = npz_data["boxes"]
            confidences = npz_data["confidences"]
            class_names = npz_data["class_name"]
            for i, fn in enumerate(frame_names):
                frames_dict[fn].append((bounding_boxes[i], confidences[i], class_names[i]))
            instance.frames = {}
            for fn, data_list in frames_dict.items():
                frame_detections = FrameDetections()
                for box_vals, conf, cls_name in data_list:
                    xtl, ytl, xbr, ybr = box_vals
                    frame_detections.detections.append(
                        BoundingBox(xtl, ytl, xbr, ybr, class_name=cls_name, confidence=conf)
                    )
                instance.frames[fn] = frame_detections

        return instance

    def get_frame_detections(self, frame_name: str) -> FrameDetections or None:
        return self.frames.get(frame_name)

    def get_all_bbox_in_frame(self, frame_name: str) -> list[list]:
        frame = self.get_frame_detections(frame_name)
        if frame:
            return [detection.get_tlbr() for detection in frame.detections]
        return []

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
    detection_path = os.path.join(closed_path, video_name, "object_detection.npz")
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
        closed_npz_path, open_npz_path = get_detection_paths(video_name, closed_path, open_path)
        # Now both read from NPZ
        json_detections = DetectionsManager.from_npz(closed_npz_path)
        npz_detections = DetectionsManager.from_npz(open_npz_path)

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
    # Now both are NPZ
    json_detections = DetectionsManager.from_npz(detection_path)
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


def count_total_boxes(detections_manager):
    """Count the total number of bounding boxes in the detections manager."""
    total_boxes = 0
    for video_name in detections_manager.keys():
        for _, detection in detections_manager[video_name].all_detections():
            total_boxes += len(detection)
    return total_boxes


def convert_json_to_npz(json_path):
    """Convert the json file to npz."""
    with open(json_path, "r") as f:
        json_data = json.load(f)
    frame_names = []
    class_name = []
    bounding_boxes = []
    confidences = []
    image_size = []
    for fn, data in json_data.get("obj_detection", {}).items():
        for detection in data["objects"]:
            frame_names.append(fn)
            class_name.append(detection["class"])
            bounding_boxes.append(detection["box"])
            confidences.append(detection["confidence"])
            image_size.append(data.get("imagesize", []))

    np.savez(
        json_path.replace(".json", ".npz"),
        frame_names=frame_names,
        class_name=class_name,
        boxes=bounding_boxes,
        confidences=confidences,
        image_size=image_size,
    )


def convert_all_json_to_npz(closed_folder):
    """Convert all JSON files in the closed folder to NPZ."""
    for root, _, files in os.walk(closed_folder):
        for file in tqdm(files, desc="Converting JSON to NPZ"):
            if file.endswith("object_detection.json"):
                json_path = os.path.join(root, file)
                convert_json_to_npz(json_path)
                print(f"Converted {json_path} to NPZ")


if __name__ == "__main__":
    visualize_detections_for_video("object_video_32.mp4")
    exit()
    save_fig = False
    plot_confidence_histograms_multiple_iou(match_multi_source_detections, save_fig=False)
    jsons_detections_manager, npz_detections_manager = match_multi_source_detections()

    # count total boxes
    total_boxes_json = count_total_boxes(jsons_detections_manager)
    total_boxes_npz = count_total_boxes(npz_detections_manager)

    print(f"total boxes in json: {total_boxes_json}")
    print(f"total boxes in npz: {total_boxes_npz}")
    plot_size_vs_confidence_per_class(jsons_detections_manager, save_fig)
    plot_matches_per_class(jsons_detections_manager, npz_detections_manager, save_fig)
    plot_confidence_histogram(jsons_detections_manager, save_fig)
    plot_confidence_histogram_per_class(jsons_detections_manager, save_fig)
    plot_box_size_histogram_per_class(jsons_detections_manager, save_fig)
