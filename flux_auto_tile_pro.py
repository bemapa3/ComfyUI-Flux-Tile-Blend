import torch
import math
import torch.nn.functional as F
import numpy as np
import cv2

def ecc_align(source_img, reference_img, warp_mode=cv2.MOTION_TRANSLATION, termination_eps=1e-7, number_of_iterations=5000):
    src_np = cv2.cvtColor((source_img[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ref_np = cv2.cvtColor((reference_img[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    
    try:
        (cc, warp_matrix) = cv2.findTransformECC(ref_np, src_np, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            return -warp_matrix[0, 2], -warp_matrix[1, 2]
        else:
            return -warp_matrix[0, 2], -warp_matrix[1, 2]
    except cv2.error as e:
        print(f"FluxAutoAligner Error: {e}. Check image clarity and content for alignment features.")
        return 0, 0

def create_smoothstep_mask(TH, TW, overlap, device):
    mask = torch.ones((TH, TW), device=device)
    if overlap > 0:
        linspace = torch.linspace(0, 1, overlap, device=device)
        fade = 3 * linspace**2 - 2 * linspace**3
        mask[:overlap, :] *= fade.view(-1, 1)
        mask[-overlap:, :] *= fade.flip(0).view(-1, 1)
        mask[:, :overlap] *= fade.view(1, -1)
        mask[:, -overlap:] *= fade.flip(0).view(1, -1)
    return mask.view(1, TH, TW, 1)

def unsharp_mask(image, kernel_size=3, strength=0.3):
    img_perm = image.permute(0, 3, 1, 2)
    blurred = F.avg_pool2d(img_perm, kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)
    high_freq = img_perm - blurred
    sharpened = img_perm + strength * high_freq
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
        padded_W = (nx - 1) * stride + tile_size
        padded_H = (ny - 1) * stride + tile_size
        pad_right = max(0, padded_W - W)
        pad_bottom = max(0, padded_H - H)
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
                "sharpen_final": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "auto_align_ecc": ("BOOLEAN", {"default": False, "label_on": "Bật ECC Alignment", "label_off": "Tắt ECC Alignment"}),
                "manual_nudge_x": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "label": "Vi chỉnh Ngang X"}),
                "manual_nudge_y": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "label": "Vi chỉnh Dọc Y"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "auto_shift_x", "auto_shift_y")
    FUNCTION = "stitch_image"
    CATEGORY = "Flux/AutoTile"

    def stitch_image(self, tiles, grid_x, grid_y, orig_width, orig_height, overlap, sharpen_final, reference_image=None, auto_align_ecc=False, manual_nudge_x=0, manual_nudge_y=0):
        B, TH, TW, C = tiles.shape
        stride_x = TW - overlap
        stride_y = TH - overlap
        padded_W = (grid_x - 1) * stride_x + TW
        padded_H = (grid_y - 1) * stride_y + TH
        output = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        weight_map = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
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
        final_image = final_image[:, :orig_height, :orig_width, :]
        
        shift_x, shift_y = 0, 0
        if auto_align_ecc and reference_image is not None:
            shift_x, shift_y = ecc_align(final_image, reference_image)
            final_shift_x = int(round(shift_x)) + manual_nudge_x
            final_shift_y = int(round(shift_y)) + manual_nudge_y
        else:
            final_shift_x = manual_nudge_x
            final_shift_y = manual_nudge_y

        if final_shift_x != 0 or final_shift_y != 0:
            final_image = torch.roll(final_image, shifts=(final_shift_y, final_shift_x), dims=(1, 2))
            
            if final_shift_x > 0:
                final_image[:, :, :final_shift_x, :] = final_image[:, :, final_shift_x:final_shift_x+1, :]
            elif final_shift_x < 0:
                final_image[:, :, final_shift_x:, :] = final_image[:, :, final_shift_x-1:final_shift_x, :]
            if final_shift_y > 0:
                final_image[:, :final_shift_y, :, :] = final_image[:, final_shift_y:final_shift_y+1, :, :]
            elif final_shift_y < 0:
                final_image[:, final_shift_y:, :, :] = final_image[:, final_shift_y-1:final_shift_y, :, :]

        if sharpen_final > 0:
            final_image = unsharp_mask(final_image, kernel_size=3, strength=sharpen_final)
        
        return (final_image, final_shift_x, final_shift_y)

NODE_CLASS_MAPPINGS = {
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher": FluxAutoStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoTiler": "Tự động chia Tile (Flux)",
    "FluxAutoStitcher": "Tự động ghép Tile (Flux)"
}
