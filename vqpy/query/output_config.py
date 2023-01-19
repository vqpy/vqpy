from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputConfig:
    """
    The config for Query output.
    :param output_frame_vobj_num: Default as False. whether to add the number
        of the filtered vobjs as an output for each frame. If true, we will
        generate a "vobj_num" field in each frame output.
    :param output_total_vobj_num: Default as False. whether to add the number
        of the filtered vobjs as an output for the whole video. If true, we
         will generate a "total_vobj_num" field in the output.
    """
    output_frame_vobj_num: bool = False
    output_total_vobj_num: bool = False
