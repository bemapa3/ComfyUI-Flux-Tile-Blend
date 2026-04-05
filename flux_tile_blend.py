import torch

class FluxHorizontalBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_left": ("IMAGE",),
                "image_right": ("IMAGE",),
                "overlap_pixels": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_horizontal"
    CATEGORY = "Flux/TileUtils"

    def merge_horizontal(self, image_left, image_right, overlap_pixels):
        # ComfyUI image shape: [Batch, Height, Width, Channels]
        B, H, W_left, C = image_left.shape
        _, _, W_right, _ = image_right.shape

        # Tạo mask chuyển sắc (Gradient) từ 1.0 -> 0.0 theo chiều ngang
        mask = torch.linspace(1.0, 0.0, overlap_pixels).view(1, 1, overlap_pixels, 1).to(image_left.device)

        # Trích xuất vùng chồng lấn (overlap) của 2 ảnh
        overlap_left = image_left[:, :, -overlap_pixels:, :]
        overlap_right = image_right[:, :, :overlap_pixels, :]

        # Hòa trộn mềm (Alpha Blending)
        blended_overlap = overlap_left * mask + overlap_right * (1.0 - mask)

        # Ghép 3 phần: Trái + Vùng hòa trộn + Phải
        left_part = image_left[:, :, :-overlap_pixels, :]
        right_part = image_right[:, :, overlap_pixels:, :]
        final_image = torch.cat((left_part, blended_overlap, right_part), dim=2)

        return (final_image,)

class FluxVerticalBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_top": ("IMAGE",),
                "image_bottom": ("IMAGE",),
                "overlap_pixels": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_vertical"
    CATEGORY = "Flux/TileUtils"

    def merge_vertical(self, image_top, image_bottom, overlap_pixels):
        B, H_top, W, C = image_top.shape
        _, H_bottom, _, _ = image_bottom.shape

        # Tạo mask chuyển sắc từ 1.0 -> 0.0 theo chiều dọc
        mask = torch.linspace(1.0, 0.0, overlap_pixels).view(1, overlap_pixels, 1, 1).to(image_top.device)

        overlap_top = image_top[:, -overlap_pixels:, :, :]
        overlap_bottom = image_bottom[:, :overlap_pixels, :, :]

        blended_overlap = overlap_top * mask + overlap_bottom * (1.0 - mask)

        top_part = image_top[:, :-overlap_pixels, :, :]
        bottom_part = image_bottom[:, overlap_pixels:, :, :]
        final_image = torch.cat((top_part, blended_overlap, bottom_part), dim=1)

        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "FluxHorizontalBlend": FluxHorizontalBlend,
    "FluxVerticalBlend": FluxVerticalBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxHorizontalBlend": "Khâu viền Tile (Ngang)",
    "FluxVerticalBlend": "Khâu viền Tile (Dọc)"
}
