import torch
import math
import torch.nn.functional as F

# Helper function to create high-order Smoothstep mask for v3.0 blend
def create_smoothstep_mask(TH, TW, overlap, device):
    mask = torch.ones((TH, TW), device=device)
    if overlap > 0:
        linspace = torch.linspace(0, 1, overlap, device=device)
        # 3.0 Blend: Smoothstep (3x^2 - 2x^3) - Sharper than Cosine, still invisible
        fade = 3 * linspace**2 - 2 * linspace**3
        
        mask[:overlap, :] *= fade.view(-1, 1)
        mask[-overlap:, :] *= fade.flip(0).view(-1, 1)
        mask[:, :overlap] *= fade.view(1, -1)
        mask[:, -overlap:] *= fade.flip(0).view(1, -1)
    
    return mask.view(1, TH, TW, 1) # Expand once, avoid repeat ops

# Edge-preserving unsharp mask for post-process sharpening
def unsharp_mask(image, kernel_size=3, strength=0.3):
    # Image is [B, H, W, C]. Must permute for F.avg_pool2d which is B, C, H, W
    img_perm = image.permute(0, 3, 1, 2)
    # Average blur to create local mean
    blurred = F.avg_pool2d(img_perm, kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)
    # Residual high-frequency detail
    high_freq = img_perm - blurred
    # Add a percentage of high freq detail back to image
    sharpened = img_perm + strength * high_freq
    # Trả về trục cũ
    return sharpened.permute(0, 2, 3, 1).clamp(0, 1)

class FluxAutoTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("tiles", "grid_x", "grid_y", "orig_width", "orig_height")
    FUNCTION = "split_image"
    CATEGORY = "Flux/AutoTile"

    def split_image(self, image, tile_size, overlap):
        B, H, W, C = image.shape
        stride = tile_size - overlap
        
        nx = math.ceil((W - overlap) / stride) if W > tile_size else 1
        ny = math.ceil((H - overlap) / stride) if H > tile_size else 1
        
        # padding lưới ảo hoàn hảo (Padded Size)
        padded_W = (nx - 1) * stride + tile_size
        padded_H = (ny - 1) * stride + tile_size
        
        pad_right = max(0, padded_W - W)
        pad_bottom = max(0, padded_H - H)
        
        # Độn thêm pixel (Replicate)
        if pad_right > 0 or pad_bottom > 0:
            img_permuted = image.permute(0, 3, 1, 2)
            img_padded = F.pad(img_permuted, (0, pad_right, 0, pad_bottom), mode='replicate')
            image = img_padded.permute(0, 2, 3, 1)
        
        tiles = []
        for y in range(ny):
            for x in range(nx):
                tiles.append(image[:, y*stride:y*stride+tile_size, x*stride:x*stride+tile_size, :])

        return (torch.cat(tiles, dim=0), nx, ny, W, H)

class FluxAutoStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "grid_x": ("INT", {"default": 1}),
                "grid_y": ("INT", {"default": 1}),
                "orig_width": ("INT", {"default": 2048}),
                "orig_height": ("INT", {"default": 2048}),
                "overlap": ("INT", {"default": 128}),
                # NÂNG CẤP: Chỉnh độ nét hậu kỳ (Post-Process Sharpness)
                "sharpen_final": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_image"
    CATEGORY = "Flux/AutoTile"

    def stitch_image(self, tiles, grid_x, grid_y, orig_width, orig_height, overlap, sharpen_final):
        B, TH, TW, C = tiles.shape
        stride_x = TW - overlap
        stride_y = TH - overlap
        
        padded_W = (grid_x - 1) * stride_x + TW
        padded_H = (grid_y - 1) * stride_y + TH
        
        output = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        weight_map = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        
        # NÂNG CẤP: S-Curve Smoothstep Blend v3.0 cho viền cực sắc
        mask = create_smoothstep_mask(TH, TW, overlap, tiles.device).expand(1, TH, TW, C)
        
        idx = 0
        for y in range(grid_y):
            for x in range(grid_x):
                x_start = x * stride_x
                y_start = y * stride_y
                tile = tiles[idx:idx+1, :, :, :]
                output[:, y_start:y_start+TH, x_start:x_start+TW, :] += tile * mask
                weight_map[:, y_start:y_start+TH, x_start:x_start+TW, :] += mask
                idx += 1

        final_image = output / (weight_map + 1e-8)
        
        # Cắt bỏ phần rìa dư thừa
        final_image = final_image[:, :orig_height, :orig_width, :]

        # NÂNG CẤP MỚI: Áp dụng Lọc Nét Hậu Kỳ để "đanh" lại chi tiết
        if sharpen_final > 0:
            # Subtle, edge-preserving unsharp mask
            final_image = unsharp_mask(final_image, kernel_size=3, strength=sharpen_final)
        
        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher": FluxAutoStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoTiler": "Tự động chia Tile (Flux)",
    "FluxAutoStitcher": "Tự động ghép Tile (Flux)"
}
