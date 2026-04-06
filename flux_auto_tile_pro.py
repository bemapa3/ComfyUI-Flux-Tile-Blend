import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

# --- 1. NODE CHIA LƯỚI (FLUX AUTO TILER) ---
class FluxAutoTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("tiles", "grid_x", "grid_y", "orig_width", "orig_height")
    FUNCTION = "tile"
    CATEGORY = "BBB-Custom"

    def tile(self, image, tile_size, overlap):
        batch, height, width, channels = image.shape
        grid_x = math.ceil((width - overlap) / (tile_size - overlap))
        grid_y = math.ceil((height - overlap) / (tile_size - overlap))
        
        tiles = []
        for y in range(grid_y):
            for x in range(grid_x):
                y_start = y * (tile_size - overlap)
                x_start = x * (tile_size - overlap)
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                
                tile = image[:, y_start:y_end, x_start:x_end, :]
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    pad_h = tile_size - tile.shape[1]
                    pad_w = tile_size - tile.shape[2]
                    tile = F.pad(tile.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode='constant', value=0).permute(0, 2, 3, 1)
                tiles.append(tile)
        
        return (torch.cat(tiles, dim=0), grid_x, grid_y, width, height)

# --- 2. NODE GHÉP HÌNH V2 (BẢN FIX LỖI CACHE & THIẾU BIẾN) ---
class FluxAutoStitcher_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "grid_x": ("INT", {"default": 1}),
                "grid_y": ("INT", {"default": 1}),
                "orig_width": ("INT", {"default": 1024}),
                "orig_height": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "BBB-Custom"

    def stitch(self, tiles, reference_image, grid_x, grid_y, orig_width, orig_height):
        batch_size, tile_h, tile_w, channels = tiles.shape
        canvas = torch.zeros((1, orig_height, orig_width, channels), device=tiles.device)
        
        idx = 0
        h_step = (orig_height - tile_h) / (grid_y - 1) if grid_y > 1 else 0
        w_step = (orig_width - tile_w) / (grid_x - 1) if grid_x > 1 else 0
        
        for y in range(grid_y):
            for x in range(grid_x):
                if idx < batch_size:
                    y_start = int(round(y * h_step))
                    x_start = int(round(x * w_step))
                    y_end = min(y_start + tile_h, orig_height)
                    x_end = min(x_start + tile_w, orig_width)
                    canvas[0, y_start:y_end, x_start:x_end, :] = tiles[idx, 0:(y_end-y_start), 0:(x_end-x_start), :]
                    idx += 1
        return (canvas,)

# --- 3. SIÊU NODE FIX TILE (BBB FREQUENCY TILE FIX) ---
class BBB_Frequency_Tile_Fix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_detailed": ("IMAGE",),
                "image_original": ("IMAGE",),
                "blur_radius": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "detail_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fix_frequency"
    CATEGORY = "BBB-Custom"

    def fix_frequency(self, image_detailed, image_original, blur_radius, detail_boost):
        device = image_detailed.device
        img_det = image_detailed.to(torch.float32)
        img_orig = image_original.to(torch.float32)
        
        if img_det.shape != img_orig.shape:
            img_orig = F.interpolate(img_orig.permute(0, 3, 1, 2), size=(img_det.shape[1], img_det.shape[2]), mode='bilinear').permute(0, 2, 3, 1)

        img_det_t = img_det.permute(0, 3, 1, 2)
        img_orig_t = img_orig.permute(0, 3, 1, 2)
        
        padding = blur_radius
        kernel_size = blur_radius * 2 + 1
        
        padded_orig = F.pad(img_orig_t, (padding, padding, padding, padding), mode='reflect')
        low_freq = F.avg_pool2d(padded_orig, kernel_size=kernel_size, stride=1)
        
        padded_det = F.pad(img_det_t, (padding, padding, padding, padding), mode='reflect')
        low_freq_det = F.avg_pool2d(padded_det, kernel_size=kernel_size, stride=1)
        
        high_freq = img_det_t - low_freq_det
        result_t = low_freq + (high_freq * detail_boost)
        result = torch.clamp(result_t.permute(0, 2, 3, 1), 0.0, 1.0)
        
        return (result,)

# --- MAPPING HỆ THỐNG (ĐỔI TÊN ĐỂ ÉP REFRESH CACHE) ---
NODE_CLASS_MAPPINGS = {
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher_V2": FluxAutoStitcher_V2,
    "BBB_Frequency_Tile_Fix": BBB_Frequency_Tile_Fix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoTiler": "Flux Auto Tiler (Fast) 🧩",
    "FluxAutoStitcher_V2": "Flux Auto Stitcher (Siêu Cấp) 🧵",
    "BBB_Frequency_Tile_Fix": "BBB Frequency Tile Fix 🛠️"
}
