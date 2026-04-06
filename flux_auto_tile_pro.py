import torch
import torch.nn.functional as F
import math

# --- 1. NODE CHIA LƯỚI (V10) ---
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

# --- 2. NODE TRỘN (V17 - SHARP EDGE & COLOR HARMONY) ---
class BBB_Smart_Photoshop_Mixer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_base": ("IMAGE",),         
                "image_detail": ("IMAGE",),       
                "texture_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "edge_sharpness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}), # Giữ viền đanh
                "color_harmony": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}), # Giảm để tránh nhòe màu
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"
    CATEGORY = "BBB-Custom"

    def mix(self, image_base, image_detail, texture_strength, edge_sharpness, color_harmony):
        base = image_base.to(torch.float32).permute(0, 3, 1, 2)
        detail = image_detail.to(torch.float32).permute(0, 3, 1, 2)
        
        # 1. Bóc High-freq sắc nét (Laplacian)
        laplacian_kernel = torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=torch.float32, device=base.device).reshape(1,1,3,3).repeat(3,1,1,1)
        high_freq = F.conv2d(detail, laplacian_kernel, padding=1, groups=3)
        
        # 2. Tạo Edge Mask đanh (Sobel-like)
        gray_base = torch.mean(base, dim=1, keepdim=True)
        edge_x = torch.abs(gray_base[:,:,:,1:] - gray_base[:,:,:,:-1])
        edge_y = torch.abs(gray_base[:,:,1:,:] - gray_base[:,:,:-1,:])
        # Pad lại cho đủ size
        edge_mask = torch.zeros_like(gray_base)
        edge_mask[:,:,:,:-1] += edge_x
        edge_mask[:,:,:-1,:] += edge_y
        edge_mask = torch.clamp(edge_mask * 15.0, 0.0, 1.0)
        
        # 3. Trộn màu thông minh (giữ cấu trúc màu nhưng không làm nhòe viền)
        low_base = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        low_detail = F.avg_pool2d(detail, kernel_size=3, stride=1, padding=1)
        color_diff = (low_base - low_detail) * color_harmony
        
        # 4. CHỐT: Đắp chi tiết nhưng "né" cái viền ra (Edge Preservation)
        # Chỗ nào có cạnh (edge_mask cao) thì giảm bớt detail đè lên để viền đanh chuẩn 
        detail_to_add = high_freq * texture_strength * (1.0 - edge_mask * edge_sharpness)
        
        result = base + detail_to_add + color_diff
        return (torch.clamp(result, 0.0, 1.0).permute(0, 2, 3, 1),)

# --- Các node Stitcher và Frequency Fix giữ nguyên như V16 ---
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
                    valid_h, valid_w = y_end - y_start, x_end - x_start
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
    FUNCTION = "fix"
    CATEGORY = "BBB-Custom"
    def fix(self, image_detailed, image_original, blur_radius, detail_boost):
        img_det = image_detailed.to(torch.float32).permute(0, 3, 1, 2)
        img_orig = image_original.to(torch.float32).permute(0, 3, 1, 2)
        H, W = img_det.shape[2], img_det.shape[3]
        scale = 8
        small_h, small_w, s_rad = max(1, H//scale), max(1, W//scale), max(1, blur_radius//scale)
        s_orig = F.interpolate(img_orig, size=(small_h, small_w), mode='bilinear')
        l_small = F.avg_pool2d(s_orig, kernel_size=s_rad*2+1, stride=1, padding=s_rad)
        l_freq = F.interpolate(l_small, size=(H, W), mode='bilinear')
        s_det = F.interpolate(img_det, size=(small_h, small_w), mode='bilinear')
        ld_small = F.avg_pool2d(s_det, kernel_size=s_rad*2+1, stride=1, padding=s_rad)
        ld_freq = F.interpolate(ld_small, size=(H, W), mode='bilinear')
        high = img_det - ld_freq
        return (torch.clamp(l_freq + (high * detail_boost), 0.0, 1.0).permute(0, 2, 3, 1),)

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
    "BBB_Smart_Photoshop_Mixer": "Mixer V17 - RAZOR SHARP 🎨"
}
