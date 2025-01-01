import cv2


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
