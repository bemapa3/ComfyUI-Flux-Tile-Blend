import torch
import math
import torch.nn.functional as F

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
        
        # Tính số lượng tile cần thiết (nx, ny)
        nx = math.ceil((W - overlap) / stride) if W > tile_size else 1
        ny = math.ceil((H - overlap) / stride) if H > tile_size else 1
        
        # Tính toán kích thước lưới ảo hoàn hảo (Padded Size)
        padded_W = (nx - 1) * stride + tile_size
        padded_H = (ny - 1) * stride + tile_size
        
        # Tính lượng pixel bị thiếu ở cạnh cần bù thêm
        pad_right = max(0, padded_W - W)
        pad_bottom = max(0, padded_H - H)
        
        # Độn thêm pixel (Replicate: copy kéo dài pixel rìa để không bị viền đen)
        if pad_right > 0 or pad_bottom > 0:
            img_permuted = image.permute(0, 3, 1, 2) # Đổi trục Tensor để F.pad hiểu
            img_padded = F.pad(img_permuted, (0, pad_right, 0, pad_bottom), mode='replicate')
            image = img_padded.permute(0, 2, 3, 1) # Trả về trục cũ
        
        tiles = []
        for y in range(ny):
            for x in range(nx):
                x_start = x * stride
                y_start = y * stride
                x_end = x_start + tile_size
                y_end = y_start + tile_size
                
                tile = image[:, y_start:y_end, x_start:x_end, :]
                tiles.append(tile)

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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_image"
    CATEGORY = "Flux/AutoTile"

    def stitch_image(self, tiles, grid_x, grid_y, orig_width, orig_height, overlap):
        B, TH, TW, C = tiles.shape
        stride_x = TW - overlap
        stride_y = TH - overlap
        
        # Khôi phục lưới ảo
        padded_W = (grid_x - 1) * stride_x + TW
        padded_H = (grid_y - 1) * stride_y + TH
        
        output = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        weight_map = torch.zeros((1, padded_H, padded_W, C), device=tiles.device)
        
        # NÂNG CẤP: Dùng S-Curve (Cosine) Gradient thay vì Linear để viền cực mượt
        mask = torch.ones((TH, TW), device=tiles.device)
        if overlap > 0:
            linspace = torch.linspace(0, 1, overlap, device=tiles.device)
            fade = (1 - torch.cos(linspace * math.pi)) / 2  # Cosine Smoothstep
            
            mask[:overlap, :] *= fade.view(-1, 1)
            mask[-overlap:, :] *= fade.flip(0).view(-1, 1)
            mask[:, :overlap] *= fade.view(1, -1)
            mask[:, -overlap:] *= fade.flip(0).view(1, -1)
        
        mask = mask.view(1, TH, TW, 1).expand(1, TH, TW, C)
        
        idx = 0
        for y in range(grid_y):
            for x in range(grid_x):
                x_start = x * stride_x
                y_start = y * stride_y
                x_end = x_start + TW
                y_end = y_start + TH
                
                if idx < tiles.shape[0]:
                    tile = tiles[idx:idx+1, :, :, :]
                    output[:, y_start:y_end, x_start:x_end, :] += tile * mask
                    weight_map[:, y_start:y_end, x_start:x_end, :] += mask
                idx += 1

        final_image = output / (weight_map + 1e-8)
        
        # Cắt bỏ phần rìa dư thừa để trả về đúng kích thước chuẩn 6000px
        final_image = final_image[:, :orig_height, :orig_width, :]
        
        return (final_image,)
