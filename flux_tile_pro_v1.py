"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         FLUX TILE PRO — Custom Nodes for ComfyUI                           ║
║         Version 1.0  |  Designed for Flux2 Klein 9B (1500-2048px optimal)  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  NODES:                                                                    ║
║    1. BBB_FluxTiler      — Cắt ảnh lớn thành tiles                        ║
║    2. BBB_FluxStitcher   — Nối tiles, cosine blend, không ghost            ║
║    3. BBB_ColorMatch     — Khớp màu về ảnh gốc                            ║
║    4. BBB_SeamRemover    — Xóa đường seam còn sót                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                                 ║
║                                                                            ║
║  [Load Image]                                                              ║
║       ↓                                                                    ║
║  [Upscale Model] → [Image Upscale x4]  ← Upscale lên 5000-6000px         ║
║       ↓ (ảnh lớn)                                                          ║
║  [BBB_FluxTiler]  ─── tiles ──────────────────────────────────────┐       ║
║       ↓ grid_info (INT)                                           ↓       ║
║  [VAEEncode] → [KSampler denoise 0.25-0.4] → [VAEDecode]                  ║
║                              ↓ tiles re-rendered                           ║
║  [BBB_FluxStitcher] ← grid_info ← từ Tiler                                ║
║       ↓ stitched image                                                     ║
║  [BBB_ColorMatch] ← original_image ← ảnh GỐC trước upscale               ║
║       ↓                                                                    ║
║  [BBB_SeamRemover] ← original_image ← ảnh GỐC (guide)                    ║
║       ↓                                                                    ║
║  [Save Image]                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TIP KSampler:
  denoise: 0.25 - 0.35  (thấp = giữ cấu trúc gốc, ít seam)
  steps:   20 - 25
  cfg:     4 - 6
  sampler: dpm++ 2m karras  hoặc  euler a
  prompt:  mô tả content (sky, stone wall, wood, v.v.)
"""

import torch
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_ramp(n: int, device) -> torch.Tensor:
    """Ramp 0→1 theo cosine, dài n pixel."""
    if n <= 0:
        return torch.zeros(0, device=device)
    return (1 - torch.cos(torch.linspace(0, math.pi, n, device=device))) * 0.5


def _make_tile_mask(tile_h: int, tile_w: int,
                    ov_top: int, ov_bottom: int,
                    ov_left: int, ov_right: int,
                    device) -> torch.Tensor:
    """
    Tạo weight mask [H, W] cho 1 tile.
    - Vùng trung tâm: weight = 1.0
    - Vùng overlap cạnh: fade-in cosine → không ghost
    - Corner: product của 2 ramp → smooth corner
    """
    mask = torch.ones((tile_h, tile_w), dtype=torch.float32, device=device)

    if ov_top > 0:
        mask[:ov_top, :] *= _cosine_ramp(ov_top, device).unsqueeze(1)
    if ov_bottom > 0 and ov_bottom <= tile_h:
        mask[-ov_bottom:, :] *= _cosine_ramp(ov_bottom, device).flip(0).unsqueeze(1)
    if ov_left > 0:
        mask[:, :ov_left] *= _cosine_ramp(ov_left, device).unsqueeze(0)
    if ov_right > 0 and ov_right <= tile_w:
        mask[:, -ov_right:] *= _cosine_ramp(ov_right, device).flip(0).unsqueeze(0)

    return mask


def _safe_pad_tile(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pad [1,H,W,C] → [1,target_h,target_w,C]. Dùng reflect nếu được."""
    h, w = t.shape[1], t.shape[2]
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return t
    t4 = t.permute(0, 3, 1, 2)  # [1,C,H,W]
    mode = "reflect" if (h > 1 and w > 1 and pad_h < h and pad_w < w) else "replicate"
    t4 = F.pad(t4, (0, pad_w, 0, pad_h), mode=mode)
    return t4.permute(0, 2, 3, 1)


def _compute_grid(width: int, height: int, tile_size: int, overlap: int):
    """Trả về (grid_x, grid_y, step_x, step_y)."""
    step = max(tile_size - overlap, tile_size // 2)
    gx = max(1, math.ceil((width  - overlap) / step))
    gy = max(1, math.ceil((height - overlap) / step))
    return gx, gy, step, step


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: BBB_FluxTiler
# ─────────────────────────────────────────────────────────────────────────────

class BBB_FluxTiler:
    """
    ✂ BBB Flux Tiler
    ─────────────────
    Cắt ảnh lớn thành tiles vuông đều nhau để đưa vào KSampler.

    OUTPUT:
      tiles          → kết nối VAEEncode
      grid_x, grid_y → kết nối BBB_FluxStitcher
      orig_w, orig_h → kết nối BBB_FluxStitcher
      tile_size      → kết nối BBB_FluxStitcher
      overlap        → kết nối BBB_FluxStitcher
    """

    OVERLAP_PRESETS = [
        "auto-balanced (~6%)",
        "small (~4%)",
        "medium (~8%)",
        "large (~12%)",
        "manual",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {
                    "default": 1536,
                    "min": 512, "max": 4096, "step": 64,
                    "tooltip": (
                        "Kích thước tile. Flux2 Klein 9B: 1536-2048px tốt nhất. "
                        "SDXL: 1024px. SD1.5: 512-768px."
                    ),
                }),
                "overlap_preset": (cls.OVERLAP_PRESETS, {"default": "auto-balanced (~6%)"}),
                "manual_overlap": ("INT", {
                    "default": 128, "min": 0, "max": 768, "step": 8,
                    "tooltip": "Chỉ dùng khi chọn 'manual'.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES  = ("tiles", "grid_x", "grid_y", "orig_w", "orig_h", "tile_size", "overlap")
    FUNCTION      = "tile"
    CATEGORY      = "BBB/TilePro"
    OUTPUT_TOOLTIPS = (
        "Batch N tiles [N,H,W,C] → VAEEncode",
        "Số cột tile → BBB_FluxStitcher",
        "Số hàng tile → BBB_FluxStitcher",
        "Width ảnh gốc → BBB_FluxStitcher",
        "Height ảnh gốc → BBB_FluxStitcher",
        "Tile size thực → BBB_FluxStitcher",
        "Overlap thực → BBB_FluxStitcher",
    )

    def tile(self, image, tile_size, overlap_preset, manual_overlap):
        _, H, W, _ = image.shape

        # Tính overlap
        if overlap_preset == "manual":
            overlap = manual_overlap
        else:
            ratio_map = {
                "auto-balanced (~6%)": 0.06,
                "small (~4%)":         0.04,
                "medium (~8%)":        0.08,
                "large (~12%)":        0.12,
            }
            raw = int(tile_size * ratio_map.get(overlap_preset, 0.06))
            overlap = max(32, (raw // 8) * 8)

        overlap = min(overlap, tile_size - 64)

        gx, gy, step_x, step_y = _compute_grid(W, H, tile_size, overlap)

        print(
            f"[BBB_FluxTiler] Input {W}×{H} → tile={tile_size}px "
            f"overlap={overlap}px grid={gx}×{gy} ({gx*gy} tiles)"
        )

        tiles = []
        for row in range(gy):
            for col in range(gx):
                y0 = row * step_y
                x0 = col * step_x
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)

                tile = image[:, y0:y1, x0:x1, :]

                # Pad nếu tile nhỏ hơn tile_size (edge tiles)
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    tile = _safe_pad_tile(tile, tile_size, tile_size)

                tiles.append(tile)

        batch = torch.cat(tiles, dim=0)   # [N, tile_size, tile_size, C]
        return (batch, gx, gy, W, H, tile_size, overlap)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: BBB_FluxStitcher
# ─────────────────────────────────────────────────────────────────────────────

class BBB_FluxStitcher:
    """
    🧵 BBB Flux Stitcher
    ─────────────────────
    Nối tiles lại thành ảnh hoàn chỉnh. Dùng cosine blend → không ghost, không nhoè.

    INPUT:
      tiles          → output của VAEDecode (sau KSampler)
      grid_x/y       → từ BBB_FluxTiler
      orig_w/h       → từ BBB_FluxTiler  (kích thước canvas đầu ra)
      tile_size      → từ BBB_FluxTiler
      overlap        → từ BBB_FluxTiler
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles":     ("IMAGE",),
                "grid_x":    ("INT", {"default": 2, "min": 1, "max": 64}),
                "grid_y":    ("INT", {"default": 2, "min": 1, "max": 64}),
                "orig_w":    ("INT", {"default": 5000, "min": 64, "max": 32768}),
                "orig_h":    ("INT", {"default": 3333, "min": 64, "max": 32768}),
                "tile_size": ("INT", {"default": 1536, "min": 64, "max": 8192}),
                "overlap":   ("INT", {"default": 128,  "min": 0,  "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stitched_image",)
    FUNCTION     = "stitch"
    CATEGORY     = "BBB/TilePro"

    def stitch(self, tiles, grid_x, grid_y, orig_w, orig_h, tile_size, overlap):
        N = tiles.shape[0]
        dev = tiles.device

        # Auto-detect actual tile size (handles nếu VAE resize tiles)
        decoded_h = tiles.shape[1]
        decoded_w = tiles.shape[2]
        C = tiles.shape[3]

        # Scale factor nếu tile size thay đổi sau VAE decode
        scale_h = decoded_h / tile_size
        scale_w = decoded_w / tile_size

        real_ov_h = int(round(overlap * scale_h))
        real_ov_w = int(round(overlap * scale_w))
        step_h    = max(decoded_h - real_ov_h, decoded_h // 2)
        step_w    = max(decoded_w - real_ov_w, decoded_w // 2)

        # Canvas kích thước đúng với orig_w × orig_h (đã scale nếu cần)
        canvas_h = int(round(orig_h * scale_h))
        canvas_w = int(round(orig_w * scale_w))

        canvas  = torch.zeros((1, canvas_h, canvas_w, C), dtype=torch.float32, device=dev)
        weights = torch.zeros((1, canvas_h, canvas_w, 1), dtype=torch.float32, device=dev)

        idx = 0
        for row in range(grid_y):
            for col in range(grid_x):
                if idx >= N:
                    break

                y0 = row * step_h
                x0 = col * step_w
                y1 = min(y0 + decoded_h, canvas_h)
                x1 = min(x0 + decoded_w, canvas_w)
                vh = y1 - y0
                vw = x1 - x0

                # Các cạnh overlap
                ot = real_ov_h if row > 0 else 0
                ob = real_ov_h if row < grid_y - 1 else 0
                ol = real_ov_w if col > 0 else 0
                or_ = real_ov_w if col < grid_x - 1 else 0

                # Clamp overlap để không vượt quá tile size
                ot = min(ot, vh);  ob = min(ob, vh)
                ol = min(ol, vw);  or_ = min(or_, vw)

                mask = _make_tile_mask(vh, vw, ot, ob, ol, or_, dev)  # [H, W]
                mask = mask.unsqueeze(0).unsqueeze(-1)                  # [1, H, W, 1]

                tile_content = tiles[idx, :vh, :vw, :].unsqueeze(0)   # [1, H, W, C]

                canvas [:, y0:y1, x0:x1, :] += tile_content * mask
                weights[:, y0:y1, x0:x1, :] += mask

                idx += 1

        # Normalize (weighted average)
        result = canvas / torch.clamp(weights, min=1e-8)
        result = torch.clamp(result, 0.0, 1.0)

        print(
            f"[BBB_FluxStitcher] {grid_x}×{grid_y} tiles → {canvas_w}×{canvas_h}px "
            f"(tile={decoded_h}px ov={real_ov_h}px)"
        )
        return (result,)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: BBB_ColorMatch
# ─────────────────────────────────────────────────────────────────────────────

class BBB_ColorMatch:
    """
    🎨 BBB Color Match
    ───────────────────
    Khớp màu sắc ảnh sau stitch về gần với ảnh gốc.

    Dùng mean+std matching theo từng channel (Lab-inspired).
    Giúp fix color drift xảy ra khi KSampler render lại từng tile.

    INPUT:
      stitched      → ảnh sau BBB_FluxStitcher
      reference     → ảnh GỐC (trước khi Upscale + Tile)
      strength      → 1.0 = copy hoàn toàn màu gốc | 0.0 = giữ nguyên
      preserve_lum  → giữ luminance của ảnh stitched (chỉ transfer màu)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitched":     ("IMAGE",),
                "reference":    ("IMAGE",),
                "strength":     ("FLOAT", {
                    "default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "preserve_lum": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Giữ luminance của stitched, chỉ chuyển màu từ reference.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("color_matched",)
    FUNCTION     = "match"
    CATEGORY     = "BBB/TilePro"

    def match(self, stitched, reference, strength, preserve_lum):
        st = stitched.to(torch.float32).permute(0, 3, 1, 2)   # [1,C,H,W]
        ref = reference.to(torch.float32).permute(0, 3, 1, 2)  # [1,C,H,W]

        dev = st.device
        ref = ref.to(dev)

        # Resize reference → same size as stitched
        if ref.shape[2:] != st.shape[2:]:
            ref = F.interpolate(ref, size=st.shape[2:], mode="bilinear", align_corners=False)

        result = torch.zeros_like(st)
        for c in range(st.shape[1]):
            sc = st[:, c:c+1]
            rc = ref[:, c:c+1]
            sm, ss = sc.mean(), sc.std().clamp(min=1e-8)
            rm, rs_ = rc.mean(), rc.std().clamp(min=1e-8)
            matched = (sc - sm) / ss * rs_ + rm
            result[:, c:c+1] = matched

        if preserve_lum:
            # Giữ luminance của stitched, chỉ lấy a/b (màu) từ matched
            # Dùng approx luminance từ RGB
            lum_st  = 0.2126*st[:, 0:1]  + 0.7152*st[:, 1:2]  + 0.0722*st[:, 2:3]
            lum_res = 0.2126*result[:, 0:1] + 0.7152*result[:, 1:2] + 0.0722*result[:, 2:3]
            # Scale result để giữ lum gốc
            scale = (lum_st + 1e-8) / (lum_res + 1e-8)
            scale = scale.clamp(0.2, 5.0)
            result = result * scale

        out = st * (1 - strength) + result * strength
        out = torch.clamp(out, 0.0, 1.0).permute(0, 2, 3, 1)
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: BBB_SeamRemover
# ─────────────────────────────────────────────────────────────────────────────

class BBB_SeamRemover:
    """
    🩹 BBB Seam Remover
    ────────────────────
    Xóa/giảm đường seam (ranh giới tiles) còn sót sau khi stitch.

    CÁCH HOẠT ĐỘNG:
      1. Phát hiện seam positions từ grid_x, grid_y, tile_size, overlap
      2. Tại mỗi seam: lấy patch từ ORIGINAL ảnh (upscaled) → blur → blend
         → xóa đường cứng, giữ texture từ original
      3. Không dùng blur toàn ảnh → chỉ xử lý vùng seam

    INPUT:
      stitched       → ảnh sau BBB_FluxStitcher
      original       → ảnh đã upscale (đưa vào Tiler) — dùng làm guide texture
      grid_x/y       → từ BBB_FluxTiler
      tile_size      → từ BBB_FluxTiler
      overlap        → từ BBB_FluxTiler
      seam_width     → bề rộng vùng xử lý mỗi bên seam (px)
      blend_strength → mức độ blend với original (0=giữ stitched, 1=lấy original)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitched":       ("IMAGE",),
                "original":       ("IMAGE",),
                "grid_x":         ("INT", {"default": 2, "min": 1, "max": 64}),
                "grid_y":         ("INT", {"default": 2, "min": 1, "max": 64}),
                "tile_size":      ("INT", {"default": 1536, "min": 64, "max": 8192}),
                "overlap":        ("INT", {"default": 128, "min": 0, "max": 2048}),
                "seam_width":     ("INT", {
                    "default": 32, "min": 4, "max": 256, "step": 4,
                    "tooltip": "Bề rộng vùng seam xử lý mỗi bên (px). Lớn hơn = mượt hơn nhưng có thể làm mờ detail.",
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0.0 = giữ nguyên stitched | 1.0 = dùng hoàn toàn từ original upscaled",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("seam_fixed",)
    FUNCTION     = "remove_seams"
    CATEGORY     = "BBB/TilePro"

    def remove_seams(self, stitched, original, grid_x, grid_y, tile_size, overlap,
                     seam_width, blend_strength):
        st  = stitched.to(torch.float32).permute(0, 3, 1, 2)   # [1,C,H,W]
        ori = original.to(torch.float32).permute(0, 3, 1, 2)   # [1,C,H,W]
        dev = st.device
        ori = ori.to(dev)

        # Resize original → same size as stitched
        if ori.shape[2:] != st.shape[2:]:
            ori = F.interpolate(ori, size=st.shape[2:], mode="bilinear", align_corners=False)

        _, C, H, W = st.shape
        result = st.clone()
        step = max(tile_size - overlap, tile_size // 2)

        # Gaussian-like 1D kernel cho seam blend
        half = seam_width
        ramp = _cosine_ramp(half, dev)               # 0→1
        full_ramp = torch.cat([ramp, ramp.flip(0)])  # 0→1→0, peak ở giữa

        def apply_seam_blend(axis: str, positions):
            """Blend dọc theo seam tại từng position."""
            nonlocal result
            for pos in positions:
                lo = max(0, pos - half)
                hi = min((H if axis == 'h' else W), pos + half)
                pw = hi - lo   # patch width

                if pw <= 0:
                    continue

                # Build blend mask [1, 1, H, W]
                if axis == 'h':
                    local_ramp = full_ramp[:pw].view(1, 1, pw, 1).expand(1, 1, pw, W)
                    seam_st  = result[:, :, lo:hi, :]
                    seam_ori = ori   [:, :, lo:hi, :]
                    blended  = seam_st * (1 - blend_strength * local_ramp) + \
                               seam_ori * blend_strength * local_ramp
                    result[:, :, lo:hi, :] = blended
                else:
                    local_ramp = full_ramp[:pw].view(1, 1, 1, pw).expand(1, 1, H, pw)
                    seam_st  = result[:, :, :, lo:hi]
                    seam_ori = ori   [:, :, :, lo:hi]
                    blended  = seam_st * (1 - blend_strength * local_ramp) + \
                               seam_ori * blend_strength * local_ramp
                    result[:, :, :, lo:hi] = blended

        # Tính vị trí seam theo X (vertical seams)
        seam_x = [step * col for col in range(1, grid_x)]
        # Tính vị trí seam theo Y (horizontal seams)
        seam_y = [step * row for row in range(1, grid_y)]

        apply_seam_blend('v', seam_x)
        apply_seam_blend('h', seam_y)

        # Soft denoise tại seam zones để giảm aliasing mà không blur toàn ảnh
        result = torch.clamp(result, 0.0, 1.0).permute(0, 2, 3, 1)
        return (result,)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRATIONS
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BBB_FluxTiler":    BBB_FluxTiler,
    "BBB_FluxStitcher": BBB_FluxStitcher,
    "BBB_ColorMatch":   BBB_ColorMatch,
    "BBB_SeamRemover":  BBB_SeamRemover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBB_FluxTiler":    "✂ BBB Flux Tiler (cắt ảnh lớn)",
    "BBB_FluxStitcher": "🧵 BBB Flux Stitcher (nối tiles, cosine blend)",
    "BBB_ColorMatch":   "🎨 BBB Color Match (khớp màu về gốc)",
    "BBB_SeamRemover":  "🩹 BBB Seam Remover (xóa đường seam)",
}
