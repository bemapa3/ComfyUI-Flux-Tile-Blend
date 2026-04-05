import torch
import math
import torch.nn.functional as F
import numpy as np
import cv2

def advanced_align(source_img, reference_img):
    """
    Titan Aligner v6.0: Kết hợp Gaussian Blur và Phase Correlation 
    để trị dứt điểm lỗi bóng ma trên ảnh Archviz.
    """
    # 1. Convert sang Numpy Gray
    src_np = (source_img[0].cpu().numpy() * 255).astype(np.uint8)
    ref_np = (reference_img[0].cpu().numpy() * 255).astype(np.uint8)
    src_gray = cv2.cvtColor(src_np, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)

    # 2. Làm mịn ảnh (Gaussian Blur) để AI không bị lừa bởi mấy chi tiết nhiễu của AI render
    src_blur = cv2.GaussianBlur(src_gray, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    try:
        # 3. Sử dụng Phase Correlation để tìm độ lệch Displacement cực nhanh và chính xác
        # Thuật toán này cực mạnh trong việc xử lý lệch lớn (như hình ní bị)
        shift, response = cv2.phaseCorrelate(ref_blur.astype(np.float32), src_blur.astype(np.float32))
        sx, sy = shift
        
        # Kiểm tra nếu kết quả quá ảo (vượt quá 50px) thì dùng ECC cứu cánh
        if abs(sx) > 50 or abs(sy) > 50:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-5)
            _, warp_matrix = cv2.findTransformECC(ref_blur, src_blur, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
            sx, sy = warp_matrix[0, 2], warp_matrix[1, 2]

        return sx, sy
    except:
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
        padded_W, padded_H = (nx - 1) * stride + tile_size, (ny - 1) * stride + tile_size
        pad_right, pad_bottom = max(0, padded_W - W), max(0, padded_H - H)
        if pad_right > 0 or pad_bottom > 0:
            image = F.pad(image.permute(0, 3, 1, 2), (0, pad_right, 0, pad_bottom), mode='replicate').permute(0, 2, 3, 1)
        tiles = [image[:, y*stride:y*stride+tile_size, x*stride:x*stride+tile_size, :] for y in range(ny) for x in range(nx)]
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
                "auto_align": ("BOOLEAN", {"default": True, "label_on": "Bật Titan Align v6.0", "label_off": "Tắt Align"}),
                "manual_nudge_x": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "manual_nudge_y": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "shift_x", "shift_y")
    FUNCTION = "stitch_image"
    CATEGORY = "Flux/AutoTile"

    def stitch_image(self, tiles, grid_x, grid_y, orig_width, orig_height, overlap, sharpen_final, reference_image=None, auto_align=True, manual_nudge_x=0, manual_nudge_y=0):
        B, TH, TW, C = tiles.shape
        stride_x, stride_y = TW - overlap, TH - overlap
        padded_W, padded_H = (grid_x - 1) * stride_x + TW, (grid_y - 1) * stride_y + TH
        output = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        weight_map = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        mask = create_smoothstep_mask(TH, TW, overlap, tiles.device).expand(1, TH, TW, C)
        
        idx = 0
        for y in range(grid_y):
            for x in range(grid_x):
                y_s, x_s = y * stride_y, x * stride_x
                output[:, y_s:y_s+TH, x_s:x_s+TW, :] += tiles[idx:idx+1] * mask
                weight_map[:, y_s:y_s+TH, x_s:x_s+TW, :] += mask
                idx += 1

        final_image = (output / (weight_map + 1e-8))[:, :orig_height, :orig_width, :]
        
        final_shift_x, final_shift_y = manual_nudge_x, manual_nudge_y
        
        if auto_align and reference_image is not None:
            # Resize ref image để so khớp (Fix scale mismatch)
            ref_resized = F.interpolate(reference_image[0:1].permute(0, 3, 1, 2), size=(orig_height, orig_width), mode='bilinear').permute(0, 2, 3, 1)
            auto_x, auto_y = advanced_align(final_image, ref_resized)
            final_shift_x += int(round(auto_x))
            final_shift_y += int(round(auto_y))
            print(f"Titan Aligner v6.0 Applied Shift: X={final_shift_x}, Y={final_shift_y}")

        if final_shift_x != 0 or final_shift_y != 0:
            final_image = torch.roll(final_image, shifts=(final_shift_y, final_shift_x), dims=(1, 2))

        if sharpen_final > 0:
            final_image = unsharp_mask(final_image, kernel_size=3, strength=sharpen_final)
            
        return (final_image, final_shift_x, final_shift_y)

NODE_CLASS_MAPPINGS = {"FluxAutoTiler": FluxAutoTiler, "FluxAutoStitcher": FluxAutoStitcher}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxAutoTiler": "Tự động chia Tile (Flux)", "FluxAutoStitcher": "Tự động ghép Tile (Flux)"}
