import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

# --- UTILS XỬ LÝ ẢNH SIÊU NHẸ RAM ---

def advanced_align(source_img, reference_img):
    """Titan Aligner v6.0: Phase Correlation (Dùng để khớp tổng thể)"""
    # Chuyển về CPU numpy để xử lý OpenCV cho nhẹ VRAM
    src_np = (source_img[0].cpu().numpy() * 255).astype(np.uint8)
    ref_np = (reference_img[0].cpu().numpy() * 255).astype(np.uint8)
    
    src_gray = cv2.cvtColor(src_np, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
    
    # Blur nhẹ để giảm noise khi tính toán shift
    src_blur = cv2.GaussianBlur(src_gray, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    
    try:
        shift, response = cv2.phaseCorrelate(ref_blur.astype(np.float32), src_blur.astype(np.float32))
        sx, sy = shift
        # Giới hạn nếu lệch quá ảo (>50px) thì reset về 0
        if abs(sx) > 50 or abs(sy) > 50:
            sx, sy = 0, 0
        return sx, sy
    except:
        return 0, 0

# --- CLASS CHÍNH CỦA NODE BBB ---

class BBB_Frequency_Tile_Fix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_detailed": ("IMAGE",), # Ảnh sau khi ghép Tile (Nét nhưng lỗi sáng)
                "image_original": ("IMAGE",), # Ảnh gốc chưa Tile (Mờ nhưng sáng chuẩn)
                "blur_radius": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "detail_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fix_frequency"
    CATEGORY = "BBB-Custom"

    def fix_frequency(self, image_detailed, image_original, blur_radius, detail_boost):
        # Chuyển sang Float32 để tính toán không bị lỗi rác ảnh
        img_det = image_detailed.to(torch.float32)
        img_orig = image_original.to(torch.float32)
        
        # B, H, W, C -> B, C, H, W
        img_det_t = img_det.permute(0, 3, 1, 2)
        img_orig_t = img_orig.permute(0, 3, 1, 2)
        
        # --- THUẬT TOÁN TÁCH TẦN SỐ TIẾT KIỆM RAM ---
        # Dùng AvgPool thay cho Gaussian để tránh tốn 359GB RAM
        padding = blur_radius
        kernel_size = blur_radius * 2 + 1
        
        # Lấy "hồn" ánh sáng từ ảnh gốc (Low Frequency)
        # Pad reflect để tránh lỗi đen viền ảnh 5000px
        padded_orig = F.pad(img_orig_t, (padding, padding, padding, padding), mode='reflect')
        low_freq = F.avg_pool2d(padded_orig, kernel_size=kernel_size, stride=1)
        
        # Lấy "chi tiết" từ ảnh Tile (High Frequency)
        # Tương tự, pad ảnh tile để trừ cho khớp kích thước
        padded_det = F.pad(img_det_t, (padding, padding, padding, padding), mode='reflect')
        low_freq_det = F.avg_pool2d(padded_det, kernel_size=kernel_size, stride=1)
        
        high_freq = img_det_t - low_freq_det
        
        # --- MIX LẠI: Ánh sáng gốc + (Chi tiết từ Tile * Boost) ---
        result_t = low_freq + (high_freq * detail_boost)
        
        # B, C, H, W -> B, H, W, C
        result = result_t.permute(0, 2, 3, 1)
        
        # Cắt giá trị thừa để ảnh không bị cháy sáng
        return (torch.clamp(result, 0.0, 1.0),)

# --- CLASS CỦA MẤY THẰNG TILE CŨ (DỰ PHÒNG) ---

class FluxAutoTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "tile_size": ("INT", {"default": 1024}), "overlap": ("INT", {"default": 128})}}
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    FUNCTION = "tile"
    CATEGORY = "BBB-Custom"
    def tile(self, image, tile_size, overlap):
        # Logic tiler cơ bản ở đây...
        return (image, 1, 1)

class FluxAutoStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"tiles": ("IMAGE",), "reference_image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "BBB-Custom"
    def stitch(self, tiles, reference_image):
        # Logic stitcher cơ bản ở đây...
        return (reference_image,)

# --- MAPPING ĐỂ COMFYUI NHẬN DIỆN ---

NODE_CLASS_MAPPINGS = {
    "BBB_Frequency_Tile_Fix": BBB_Frequency_Tile_Fix,
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher": FluxAutoStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBB_Frequency_Tile_Fix": "BBB Frequency Tile Fix 🛠️",
    "FluxAutoTiler": "Flux Auto Tiler (Fast)",
    "FluxAutoStitcher": "Flux Auto Stitcher (Fast)"
}
