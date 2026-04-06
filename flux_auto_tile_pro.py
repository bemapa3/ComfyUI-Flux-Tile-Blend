import torch
import torch.nn.functional as F
import math

# --- 1. NODE CHIA LƯỚI (V10 - CẮT CHUẨN & XUẤT THÔNG SỐ) ---
class FluxAutoTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("tiles", "grid_x", "grid_y", "orig_width", "orig_height", "actual_tile_size", "actual_overlap")
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

        return (torch.cat(tiles, dim=0), grid_x, grid_y, width, height, tile_size, overlap)


# --- 2. NODE GHÉP HÌNH (V12 - TOÁN HỌC KHỚP 100% VỚI TILER) ---
class FluxAutoStitcher_AutoV10:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "reference_image": ("IMAGE",),
                # forceInput: True ép thành lỗ cắm, chống gõ tay sai lệch
                "tile_size": ("INT", {"forceInput": True}),
                "overlap": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "BBB-Custom"

    def stitch(self, tiles, reference_image, tile_size, overlap):
        batch_size, tile_h, tile_w, channels = tiles.shape
        batch, orig_height, orig_width, _ = reference_image.shape

        canvas = reference_image.clone()

        grid_x = math.ceil((orig_width - overlap) / (tile_size - overlap))
        grid_y = math.ceil((orig_height - overlap) / (tile_size - overlap))

        idx = 0
        for y in range(grid_y):
            for x in range(grid_x):
                if idx < batch_size:
                    # TRỞ VỀ CHÍNH ĐẠO: Dùng đúng công thức của Tiler
                    y_start = y * (tile_size - overlap)
                    x_start = x * (tile_size - overlap)
                    
                    y_end = min(y_start + tile_size, orig_height)
                    x_end = min(x_start + tile_size, orig_width)
                    
                    valid_h = y_end - y_start
                    valid_w = x_end - x_start

                    canvas[0, y_start:y_end, x_start:x_end, :] = tiles[idx, 0:valid_h, 0:valid_w, :]
                    idx += 1

        return (canvas,)


# --- 3. NODE VÁ SẸO (V8.0 - RESIZE TRƯỚC KHI BLUR ĐỂ CHỐNG NỔ RAM 359GB) ---
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
        
        B, H, W, C = img_det.shape
        img_det_t = img_det.permute(0, 3, 1, 2)
        img_orig_t = img_orig.permute(0, 3, 1, 2)

        # Thu nhỏ ảnh 8 lần để quét ánh sáng -> RAM nhẹ như tơ
        scale_factor = 8
        small_h, small_w = max(1, H // scale_factor), max(1, W // scale_factor)
        small_radius = max(1, blur_radius // scale_factor)
        kernel_size = small_radius * 2 + 1
        
        small_orig = F.interpolate(img_orig_t, size=(small_h, small_w), mode='bilinear')
        low_freq_small = F.avg_pool2d(small_orig, kernel_size=kernel_size, stride=1, padding=small_radius)
        low_freq = F.interpolate(low_freq_small, size=(H, W), mode='bilinear')

        small_det = F.interpolate(img_det_t, size=(small_h, small_w), mode='bilinear')
        low_freq_det_small = F.avg_pool2d(small_det, kernel_size=kernel_size, stride=1, padding=small_radius)
        low_freq_det = F.interpolate(low_freq_det_small, size=(H, W), mode='bilinear')

        high_freq = img_det_t - low_freq_det
        result_t = low_freq + (high_freq * detail_boost)
        
        return (torch.clamp(result_t.permute(0, 2, 3, 1), 0.0, 1.0),)

# --- MAPPING HỆ THỐNG ---
NODE_CLASS_MAPPINGS = {
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher_AutoV10": FluxAutoStitcher_AutoV10,
    "BBB_Frequency_Tile_Fix": BBB_Frequency_Tile_Fix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoTiler": "Flux Auto Tiler (Fast) V10 🧩",
    "FluxAutoStitcher_AutoV10": "Flux Auto Stitcher (Tự Động) 🧵",
    "BBB_Frequency_Tile_Fix": "BBB Frequency Tile Fix 🛠️"
}
