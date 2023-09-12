import cv2
import json
from shapely.geometry import Polygon
import numpy as np
import os


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def draw_regions(frame, regions, color, transparency=0.8):
    for region in regions:
        polygon = Polygon(region)
        int_coords = lambda x: np.array(x).round().astype(np.int32) # noqa
        exterior = [int_coords(polygon.exterior.coords)]
        # overlay = frame.copy()
        # cv2.fillPoly(overlay, exterior, color)
        # frame = cv2.addWeighted(overlay, transparency, frame,
        #                          1 - transparency, 0, frame)
        # outline with input color
        cv2.polylines(frame, exterior, True, color, 2)
    return frame


def draw_global_annotations(
    frame, global_annotations, line, width, color, transparency=0.6
):
    texts = global_annotations(line)

    # Calculate the height of the box based on the number of texts
    box_height = (
        30 * len(texts) + 20
    )  # 30 pixels per text line + 20 pixels padding

    overlay = frame.copy()

    # Draw a white rectangle on the overlay
    cv2.rectangle(
        overlay,
        (
            width - 300,
            10,
        ),
        (
            width - 10,
            10 + box_height,
        ),  # Subtract 10 to add a little padding from the right edge
        (255, 255, 255),
        -1,
    )

    # Blend the overlay with the frame to achieve the transparency effect
    cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

    # Draw texts
    for i, text in enumerate(texts):
        cv2.putText(
            frame,
            text,
            (
                width - 290,
                40 + 30 * i,
            ),  # Adjusted the starting x-coordinate for a bit of padding
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return frame


def save_output_video(
    video_path,
    query_result_path,
    output_video_path,
    query_class_name,
    sample_images=None,
    regions=None,
    vobj_annotations=None,
    global_annotations=None,
):
    """
    Note that the output_per_frame_results must be True when calling vqpy.init.
    Args:
        video_path: the path to the input video.
        query_result_path: the path to the query result.
        output_video_path: the path to the output video.
        query_class_name: the name of the query.
    """
    # read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # read query result
    frame_id = 0
    with open(query_result_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ret_val, frame = cap.read()
                line = json.loads(line)
                frame_id = line["frame_id"]
                # draw frame id

                frame = cv2.putText(
                    frame,
                    str(frame_id),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                # draw global annotations on the top right corner
                # each element of the global annoation is a new line
                if global_annotations is not None:
                    draw_global_annotations(
                        frame,
                        global_annotations,
                        line,
                        width,
                        color=(0, 0, 255),
                        transparency=0.6,
                    )

                # draw polygon regions with yellow color of transparency 0.6
                if regions is not None:
                    draw_regions(
                        frame, regions, color=(0, 255, 255), transparency=0.6
                    )

                filtered_vobjs = line[query_class_name]
                # draw filtered vobjs
                for filtered_vobj in filtered_vobjs:
                    tlbr = filtered_vobj["tlbr"]
                    tlbr = [
                        int(tlbr[0]),
                        int(tlbr[1]),
                        int(tlbr[2]),
                        int(tlbr[3]),
                    ]
                    # draw bounding box
                    track_id = filtered_vobj["track_id"]
                    color = get_color(track_id)
                    frame = cv2.rectangle(
                        frame, (tlbr[0], tlbr[1]), (tlbr[2], tlbr[3]), color, 2
                    )
                    # draw text
                    if vobj_annotations is not None:
                        text = vobj_annotations(filtered_vobj)
                        # draw text bbox and text
                        text_size = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2
                        )
                        frame = cv2.rectangle(
                            frame,
                            (tlbr[0], tlbr[1] - text_size[0][1] - 2),
                            (tlbr[0] + text_size[0][0], tlbr[1]),
                            color,
                            -1,
                        )
                        frame = cv2.putText(
                            frame,
                            text,
                            (tlbr[0], tlbr[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )
                out.write(frame)
            # output sample frame
            if sample_images is not None:
                for sample_frame_id in sample_images:
                    if sample_frame_id == frame_id:
                        cv2.imwrite(
                            os.path.join(
                                os.path.dirname(output_video_path),
                                str(sample_frame_id) + ".jpg",
                            ),
                            frame,
                        )
            frame_id += 1
    cap.release()
