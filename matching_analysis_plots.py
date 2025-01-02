from collections import defaultdict
from typing import Callable

from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def abbreviate_val(val):
    return f"{val // 1000}k" if val >= 1000 else str(val)


def plot_matches_per_class(jsons_detections_manager, npz_detections_manager, save_fig=False):

    matched_counts = defaultdict(int)
    unmatched_counts = defaultdict(int)

    for video_name, json_mgr in tqdm(jsons_detections_manager.items(), desc="Processing videos"):
        npz_mgr = npz_detections_manager[video_name]
        for frame_name in json_mgr.get_all_frame_names():
            boxes_json = json_mgr.get_frame_detections(frame_name)
            for box in boxes_json.detections:
                if box.is_matched:
                    matched_counts[box.class_name] += 1
                else:
                    unmatched_counts[box.class_name] += 1

    classes = list(set(matched_counts.keys()) | set(unmatched_counts.keys()))
    matched_vals = [matched_counts[c] for c in classes]
    unmatched_vals = [unmatched_counts[c] for c in classes]
    matched_text = [abbreviate_val(v) for v in matched_vals]
    unmatched_text = [abbreviate_val(v) for v in unmatched_vals]

    fig = go.Figure(
        data=[
            go.Bar(name="Matched", x=classes, y=matched_vals, text=matched_text, textposition="auto"),
            go.Bar(name="Unmatched", x=classes, y=unmatched_vals, text=unmatched_text, textposition="auto"),
        ]
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Class",
        yaxis_title="Count",
        title={
            "text": "Matched vs. Unmatched per Class (All Detected Boxes)",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis_type="log",
    )
    if save_fig:
        fig.write_html("matches_per_class.html")
    else:
        fig.show()
    del fig


def collect_confidence_scores(jsons_detections_manager):
    matched_scores = []
    unmatched_scores = []

    # Collect confidence scores
    for json_mgr in tqdm(jsons_detections_manager.values(), desc="Processing videos"):
        for frame_name in json_mgr.get_all_frame_names():
            boxes_json = json_mgr.get_frame_detections(frame_name)
            for box in boxes_json.detections:
                if box.is_matched:
                    matched_scores.append(box.confidence)
                else:
                    unmatched_scores.append(box.confidence)
    return matched_scores, unmatched_scores


def plot_confidence_histogram(jsons_detections_manager, save_fig=False):
    matched_scores, unmatched_scores = collect_confidence_scores(jsons_detections_manager)

    # Create histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=matched_scores, name="Matched", opacity=0.75, nbinsx=50, histnorm="probability"))

    fig.add_trace(go.Histogram(x=unmatched_scores, name="Unmatched", opacity=0.75, nbinsx=50, histnorm="probability"))

    fig.update_layout(
        barmode="overlay",
        title={
            "text": "Confidence Score Distribution (Matched vs Unmatched)",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Confidence Score",
        yaxis_title="Probability",
        bargap=0.1,
    )
    if save_fig:
        fig.write_html("confidence_histogram.html")
    else:
        fig.show()
    del fig


def plot_confidence_histogram_per_class(jsons_detections_manager, save_fig=False):

    # Dictionary to store scores per class
    matched_scores_per_class = defaultdict(list)
    unmatched_scores_per_class = defaultdict(list)

    # Collect confidence scores per class
    for json_mgr in tqdm(jsons_detections_manager.values(), desc="Processing videos"):
        for frame_name in json_mgr.get_all_frame_names():
            boxes_json = json_mgr.get_frame_detections(frame_name)
            for box in boxes_json.detections:
                if box.is_matched:
                    matched_scores_per_class[box.class_name].append(box.confidence)
                else:
                    unmatched_scores_per_class[box.class_name].append(box.confidence)

    # Get unique classes
    all_classes = sorted(set(matched_scores_per_class.keys()) | set(unmatched_scores_per_class.keys()))
    n_classes = len(all_classes)

    # Calculate grid dimensions
    n_cols = 3  # You can adjust this
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=all_classes)

    # Add histograms for each class
    for idx, class_name in enumerate(all_classes):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Add matched histogram
        fig.add_trace(
            go.Histogram(
                x=matched_scores_per_class[class_name],
                name="Matched",
                opacity=0.75,
                nbinsx=50,
                histnorm="probability",
                showlegend=True if idx == 0 else False,
                marker_color="rgba(0, 0, 255, 0.5)",
            ),
            row=row,
            col=col,
        )

        # Add unmatched histogram
        fig.add_trace(
            go.Histogram(
                x=unmatched_scores_per_class[class_name],
                name="Unmatched",
                opacity=0.75,
                nbinsx=50,
                histnorm="probability",
                showlegend=True if idx == 0 else False,
                marker_color="rgba(255, 0, 0, 0.5)",
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=1200,
        title={
            "text": "Confidence Score Distribution per Class (Matched vs Unmatched)",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        barmode="overlay",
        showlegend=True,
    )

    # Update all xaxes and yaxes
    fig.update_xaxes(title_text="Confidence Score")
    fig.update_yaxes(title_text="Probability")

    if save_fig:
        fig.write_html("confidence_histogram_per_class.html")
    else:
        fig.show()
    del fig


def plot_box_size_histogram_per_class(jsons_detections_manager, save_fig=False):

    # Dictionary to store normalized box sizes per class
    matched_sizes_per_class = defaultdict(list)
    unmatched_sizes_per_class = defaultdict(list)

    # Collect normalized box sizes per class
    for json_mgr in tqdm(jsons_detections_manager.values(), desc="Processing videos"):
        for frame_name in json_mgr.get_all_frame_names():
            boxes_json = json_mgr.get_frame_detections(frame_name)
            image_area = boxes_json.image_size[0] * boxes_json.image_size[1]

            for box in boxes_json.detections:
                # Calculate box width and height
                width = box.xbr - box.xtl
                height = box.ybr - box.ytl
                box_area = width * height
                normalized_size = box_area / image_area

                if box.is_matched:
                    matched_sizes_per_class[box.class_name].append(normalized_size)
                else:
                    unmatched_sizes_per_class[box.class_name].append(normalized_size)
    # Get unique classes
    all_classes = sorted(set(matched_sizes_per_class.keys()) | set(unmatched_sizes_per_class.keys()))
    n_classes = len(all_classes)

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=all_classes)

    # Add histograms for each class
    for idx, class_name in enumerate(all_classes):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Add matched histogram
        fig.add_trace(
            go.Histogram(
                x=matched_sizes_per_class[class_name],
                name="Matched",
                opacity=0.75,
                nbinsx=50,
                histnorm="probability",
                showlegend=True if idx == 0 else False,
                marker_color="rgba(0, 0, 255, 0.5)",
            ),
            row=row,
            col=col,
        )

        # Add unmatched histogram
        fig.add_trace(
            go.Histogram(
                x=unmatched_sizes_per_class[class_name],
                name="Unmatched",
                opacity=0.75,
                nbinsx=50,
                histnorm="probability",
                showlegend=True if idx == 0 else False,
                marker_color="rgba(255, 0, 0, 0.5)",
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=1200,
        title={
            "text": "Normalized Box Size Distribution per Class (Matched vs Unmatched)",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        barmode="overlay",
        showlegend=True,
    )

    # Update all xaxes and yaxes
    fig.update_xaxes(title_text="Normalized Box Size")
    fig.update_yaxes(title_text="Probability")
    if save_fig:
        fig.write_html("box_size_histogram_per_class.html")
    else:
        fig.show()
    del fig


def plot_size_vs_confidence_per_class(jsons_detections_manager, save_fig=False):
    # Collect data per class
    matched_sizes_per_class = defaultdict(list)
    matched_conf_per_class = defaultdict(list)
    unmatched_sizes_per_class = defaultdict(list)
    unmatched_conf_per_class = defaultdict(list)

    # Iterate through all json managers and frames
    for json_mgr in tqdm(jsons_detections_manager.values(), desc="Processing videos"):
        for frame_name in json_mgr.get_all_frame_names():
            boxes_json = json_mgr.get_frame_detections(frame_name)
            image_area = boxes_json.image_size[0] * boxes_json.image_size[1]

            for box in boxes_json.detections:
                # Calculate normalized box size
                width = box.xbr - box.xtl
                height = box.ybr - box.ytl
                box_area = width * height
                normalized_size = box_area / image_area

                if box.is_matched:
                    matched_sizes_per_class[box.class_name].append(normalized_size)
                    matched_conf_per_class[box.class_name].append(box.confidence)
                else:
                    unmatched_sizes_per_class[box.class_name].append(normalized_size)
                    unmatched_conf_per_class[box.class_name].append(box.confidence)

    # Get all unique classes
    all_classes = sorted(set(matched_sizes_per_class.keys()) | set(unmatched_sizes_per_class.keys()))
    n_classes = len(all_classes)

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=all_classes)

    # Add scatter plots for each class
    for idx, class_name in enumerate(all_classes):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Add matched scatter
        fig.add_trace(
            go.Scattergl(
                x=matched_sizes_per_class[class_name],
                y=matched_conf_per_class[class_name],
                mode="markers",
                name="Matched",
                opacity=0.75,
                showlegend=True if idx == 0 else False,
                marker=dict(color="rgba(0, 0, 255, 0.5)", size=5),
            ),
            row=row,
            col=col,
        )

        # Add unmatched scatter
        fig.add_trace(
            go.Scattergl(
                x=unmatched_sizes_per_class[class_name],
                y=unmatched_conf_per_class[class_name],
                mode="markers",
                name="Unmatched",
                opacity=0.75,
                showlegend=True if idx == 0 else False,
                marker=dict(color="rgba(255, 0, 0, 0.5)", size=5),
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=1000,
        title_text="Box Size vs Confidence per Class",
        showlegend=True,
    )

    # Update all xaxes and yaxes
    fig.update_xaxes(title_text="Normalized Box Size")
    fig.update_yaxes(title_text="Confidence Score")

    if save_fig:
        fig.write_html("size_vs_confidence_per_class.html")
    else:
        fig.show()
    del fig


def plot_confidence_histograms_multiple_iou(match_func: Callable, save_fig=False):
    """
    Plot histograms of confidence scores for different IoU thresholds in separate subplots.
    """
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    matched_scores_dict: dict[float, list[float]] = {}
    unmatched_scores_dict: dict[float, list[float]] = {}

    # Collect scores for each IoU threshold
    for iou_threshold in tqdm(iou_thresholds, desc="Processing IoU thresholds"):
        jsons_detections_manager, _ = match_func(iou_threshold)
        matched, unmatched = collect_confidence_scores(jsons_detections_manager)
        matched_scores_dict[iou_threshold] = matched
        unmatched_scores_dict[iou_threshold] = unmatched

    # Calculate number of rows and columns for subplots
    n_plots = len(iou_thresholds)
    n_cols = 3  # You can adjust this
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"IoU = {iou}" for iou in iou_thresholds])

    # Add traces for each IoU threshold
    for idx, iou_threshold in enumerate(iou_thresholds):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        fig.add_trace(
            go.Histogram(
                x=matched_scores_dict[iou_threshold],
                name=f"Matched",
                opacity=0.75,
                nbinsx=50,
                histnorm="probability",
                showlegend=True if idx == 0 else False,
                marker_color="rgba(0, 0, 255, 0.5)",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Histogram(
                x=unmatched_scores_dict[iou_threshold],
                name=f"Unmatched",
                opacity=0.75,
                nbinsx=50,
                histnorm="probability",
                marker_color="rgba(255, 0, 0, 0.5)",
                showlegend=True if idx == 0 else False,
            ),
            row=row,
            col=col,
        )

        # Update layout for each subplot
        fig.update_xaxes(title_text="Confidence Score", row=row, col=col)
        fig.update_yaxes(title_text="probability", row=row, col=col)

    # Update overall layout
    fig.update_layout(
        height=300 * n_rows,
        width=400 * n_cols,
        title={
            "text": "Confidence Score Distribution per IoU Threshold",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=True,
        barmode="overlay",
    )
    if save_fig:
        fig.write_html("confidence_histograms_multiple_iou.html")
    else:
        fig.show()
    del fig
    plot_match_unmatch_counts(matched_scores_dict, unmatched_scores_dict, save_fig)


def plot_match_unmatch_counts(matched_scores_dict, unmatched_scores_dict, save_fig=False):
    """
    Plot matched vs unmatched counts for each IoU threshold in a single plot with log scale.
    """
    iou_thresholds = list(matched_scores_dict.keys())

    # Calculate counts
    matched_counts = [len(matched_scores_dict[iou]) for iou in iou_thresholds]
    unmatched_counts = [len(unmatched_scores_dict[iou]) for iou in iou_thresholds]

    # Generate abbreviated text labels
    matched_text = [abbreviate_val(v) for v in matched_counts]
    unmatched_text = [abbreviate_val(v) for v in unmatched_counts]

    # Create figure
    fig = go.Figure()

    # Add traces with text labels
    fig.add_trace(
        go.Bar(
            x=iou_thresholds,
            y=matched_counts,
            name="Matched",
            marker_color="rgba(0, 0, 255, 0.5)",
            text=matched_text,
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=iou_thresholds,
            y=unmatched_counts,
            name="Unmatched",
            marker_color="rgba(255, 0, 0, 0.5)",
            text=unmatched_text,
            textposition="auto",
        )
    )

    # Rest of the layout remains the same
    fig.update_layout(
        title={
            "text": "Matched vs Unmatched Counts per IoU Threshold",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="IoU Threshold",
        yaxis_title="Count (log scale)",
        yaxis_type="log",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
    )

    if save_fig:
        fig.write_html("match_unmatch_counts.html")
    else:
        fig.show()
    del fig
