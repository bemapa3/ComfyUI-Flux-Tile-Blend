import os
import sys

# Đảm bảo Python tìm thấy thư mục hiện tại
sys.path.append(os.path.dirname(__file__))

# 1. Gọi Node từ file Tile Pro (Tiler, Stitcher, v.v.)
from flux_tile_pro_v1 import NODE_CLASS_MAPPINGS as pro_classes, NODE_DISPLAY_NAME_MAPPINGS as pro_names

# 2. Gọi Node từ file Tile Equalizer
from bbb_tile_equalizer_v1 import NODE_CLASS_MAPPINGS as eq_classes, NODE_DISPLAY_NAME_MAPPINGS as eq_names

# 3. Gộp cả 2 lại cho ComfyUI nhận diện
NODE_CLASS_MAPPINGS = {**pro_classes, **eq_classes}
NODE_DISPLAY_NAME_MAPPINGS = {**pro_names, **eq_names}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
