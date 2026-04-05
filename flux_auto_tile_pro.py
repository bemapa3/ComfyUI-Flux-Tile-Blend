import torch
import math

class FluxAutoTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 256, "min": 0, "max": 512, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("tiles", "grid_x", "grid_y", "orig_width", "orig_height")
    FUNCTION = "split_image"
    CATEGORY = "Flux/AutoTile"

    def split_image(self, image, tile_size, overlap):
        B, H, W, C = image.shape
        stride = tile_size - overlap
        
        # Tính số lượng tile cần thiết theo chiều dọc và ngang
        nx = math.ceil((W - overlap) / stride)
        ny = math.ceil((H - overlap) / stride)
        
        tiles = []
        for y in range(ny):
            for x in range(nx):
                x_start = x * stride
                y_start = y * stride
                x_end = min(x_start + tile_size, W)
                y_end = min(y_start + tile_size, H)
                
                # Nếu tile cuối bị thiếu (nhỏ hơn tile_size), lấy lùi lại để đủ size
                if x_end - x_start < tile_size:
                    x_start = max(0, x_end - tile_size)
                if y_end - y_start < tile_size:
                    y_start = max(0, y_end - tile_size)
                
                tile = image[:, y_start:y_end, x_start:x_end, :]
                tiles.append(tile)

        # Trả về Batch của các tile để đưa vào 1 KSampler duy nhất
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
                "overlap": ("INT", {"default": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_image"
    CATEGORY = "Flux/AutoTile"

    def stitch_image(self, tiles, grid_x, grid_y, orig_width, orig_height, overlap):
        B, TH, TW, C = tiles.shape
        # Tạo canvas trống để dán các tile vào
        output = torch.zeros((1, orig_height, orig_width, C), device=tiles.device)
        # Tạo canvas trọng số để tính toán trung bình cộng vùng overlap
        weight_map = torch.zeros((1, orig_height, orig_width, C), device=tiles.device)
        
        # Tạo mặt nạ hòa trộn (Gradient Mask)
        mask = torch.ones((TH, TW), device=tiles.device)
        if overlap > 0:
            # Tạo gradient mượt ở 4 cạnh
            fade = torch.linspace(0, 1, overlap, device=tiles.device)
            mask[:overlap, :] *= fade.view(-1, 1)
            mask[-overlap:, :] *= fade.flip(0).view(-1, 1)
            mask[:, :overlap] *= fade.view(1, -1)
            mask[:, -overlap:] *= fade.flip(0).view(1, -1)
        
        mask = mask.view(1, TH, TW, 1).expand(1, TH, TW, C)
        
        stride_x = TW - overlap
        stride_y = TH - overlap
        
        idx = 0
        for y in range(grid_y):
            for x in range(grid_x):
                x_start = x * stride_x
                y_start = y * stride_y
                x_end = x_start + TW
                y_end = y_start + TH
                
                # Xử lý trường hợp tile cuối bị lùi lại để đủ size
                if x_end > orig_width:
                    x_start = orig_width - TW
                    x_end = orig_width
                if y_end > orig_height:
                    y_start = orig_height - TH
                    y_end = orig_height
                
                tile = tiles[idx:idx+1, :, :, :]
                output[:, y_start:y_end, x_start:x_end, :] += tile * mask
                weight_map[:, y_start:y_end, x_start:x_end, :] += mask
                idx += 1

        # Tránh chia cho 0 và lấy kết quả cuối cùng
        final_image = output / (weight_map + 1e-8)
        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "FluxAutoTiler": FluxAutoTiler,
    "FluxAutoStitcher": FluxAutoStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoTiler": "Tự động chia Tile (Flux)",
    "FluxAutoStitcher": "Tự động ghép Tile (Flux)"
}
