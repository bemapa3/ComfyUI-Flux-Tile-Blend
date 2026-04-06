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


# --- 2. NODE TRỘN CHI TIẾT (SMART PHOTOSHOP MIXER - TRỘN TRƯỚC KHI GHÉP) ---
class BBB_Smart_Photoshop_Mixer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_base_low_denoise": ("IMAGE",),   # Cắm từ KSampler Denoise 0.2 (Giữ cấu trúc)
                "image_detail_high_denoise": ("IMAGE",), # Cắm từ KSampler Denoise 0.45 (Lấy chi tiết)
                "texture_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer_detail"
    CATEGORY = "BBB-Custom"

    def transfer_detail(self, image_base_low_denoise, image_detail_high_denoise, texture_strength):
        base = image_base_low_denoise.to(torch.float32).permute(0, 3, 1, 2)
        detail = image_detail_high_denoise.to(torch.float32).permute(0, 3, 1, 2)
        
        # High-pass filter để bóc chi tiết gai góc
        kernel_size = 5
        padding = kernel_size // 2
        low_freq_detail = F.avg_pool2d(detail, kernel_size=kernel_size, stride=1, padding=padding)
        high_freq_detail = detail - low_freq_detail
        
        # Đắp chi tiết lên ảnh Base
        result = base + (high_freq_detail * texture_strength)
        result = torch.clamp(result, 0.0, 1.0).permute(0, 2, 3, 1)
        return (result,)


# --- 3. NODE GHÉP HÌNH (V13 - FEATHER BLENDING - HÒA TRỘN MÉP MỀM) ---
class FluxAutoStitcher_Blend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "actual_tile_size": ("INT", {"forceInput": True}),
                "actual_overlap": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "BBB-Custom"

    def stitch(self, tiles, reference_image, actual_tile_size, actual_overlap):
        batch_size, tile_h, tile_w, channels = tiles.shape
        batch, orig_height, orig_width, _ = reference_image.shape

        canvas = torch.zeros_like(reference_image, dtype=torch.float32)
        weights = torch.zeros((1, orig_height, orig_width, 1), dtype=torch.float32, device=reference_image.device)

        grid_x = math.ceil((orig_width - actual_overlap) / (actual_tile_size - actual_overlap))
        grid_y = math.ceil((orig_height - actual_overlap) / (actual_tile_size - actual_overlap))

        idx = 0
        for y in range(grid_y):
            for x in range(grid_x):
                if idx < batch_size:
                    y_start = y * (actual_tile_size - actual_overlap)
                    x_start = x * (actual_tile_size - actual_overlap)
                    y_end = min(y_start + actual_tile_size, orig_height)
                    x_end = min(x_start + actual_tile_size, orig_width)
                    valid_h = y_end - y_start
                    valid_w = x_end - x_start

                    # Feather mask cho vùng overlap
                    mask = torch.ones((valid_h, valid_w, 1), dtype=torch.float32, device=reference_image.device)
                    if actual_overlap > 0:
                        fade = torch.linspace(0.0, 1.0, actual_overlap, device=reference_image.device)
                        if y > 0: mask[:actual_overlap, :, 0] *= fade.unsqueeze(1)
                        if y < grid_y - 1 and valid_h >= actual_overlap: mask[-actual_overlap:, :, 0] *= fade.flip(0).unsqueeze(1)
                        if x > 0: mask[:, :actual_overlap, 0] *= fade.unsqueeze(0)
                        if x < grid_x - 1 and valid_w >= actual_overlap: mask[:, -actual_overlap:, 0] *= fade.flip(0).unsqueeze(0)

                    canvas[0, y_start:y_end, x_start:x_end, :] += tiles[idx, 0:valid_h, 0:valid_w, :] * mask
                    weights[0, y_start:y_end, x_start:x_end, :] += mask
                    idx += 1

        canvas = canvas / torch.clamp(weights, min=1e-8)
        return (torch.clamp(canvas, 0.0, 1.0),)


# --- 4. NODE VÁ SẸO (V8.0 - CHỐNG NỔ RAM) ---
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
        img_det = image_detailed.to(torch.float32).permute(0, 3, 1, 2)
        img_orig = image_original.to(torch.float32).permute(0, 3, 1, 2)
        B, C, H, W = img_det.shape

        scale_factor = 8
        small_h, small_w = max(1, H // scale_factor), max(1, W // scale_factor)
        small_radius = max(1, blur_radius // scale_factor)
        kernel_size = small_radius * 2 + 1
        
        small_orig = F.interpolate(img_orig, size=(small_h, small_w), mode='bilinear')
        low_freq_small = F.avg_pool2d(small_orig, kernel_size=kernel_size, stride=1, padding=small_radius)
        low_freq = F.interpolate(low_freq_small, size=(H, W), mode='bilinear')

        small_det = F.interpolate(img_det, size=(small_h, small_w), mode='bilinear')
        low_freq_det_small = F.avg_pool2d(small_det, kernel_size=kernel_size, stride=1, padding=small_radius)
        low_freq_det = F.interpolate(low_freq_det_small, size=(H, W), mode='bilinear')

        high_freq = img_det - low_freq_det
        result = low_freq + (high_freq * detail_boost)
        return (torch.clamp(result, 0.0, 1.0).permute(0, 2, 3, 1),)


# --- MAPPING ---
NODE_CLASS_MAPPINGS = {
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher_Blend": FluxAutoStitcher_Blend,
    "BBB_Frequency_Tile_Fix": BBB_Frequency_Tile_Fix,
    "BBB_Smart_Photoshop_Mixer": BBB_Smart_Photoshop_Mixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoTiler": "Flux Auto Tiler (Fast) V10 🧩",
    "FluxAutoStitcher_Blend": "Flux Auto Stitcher (Feather Blend) 🧵",
    "BBB_Frequency_Tile_Fix": "BBB Frequency Tile Fix 🛠️",
    "BBB_Smart_Photoshop_Mixer": "BBB Smart Photoshop Mixer 🎨"
}
