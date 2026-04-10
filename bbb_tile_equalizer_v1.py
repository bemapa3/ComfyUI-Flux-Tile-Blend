"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BBB TILE EQUALIZER — ComfyUI Custom Node                           ║
║         Version 1.0  |  Per-tile color equalization trước khi stitch       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                                 ║
║                                                                            ║
║  [BBB_FluxTiler]                                                           ║
║       ├── tiles (raw, chưa render) ──────────────────────────────────────┐ ║
║       ├── grid_x / grid_y / tile_size / overlap                          │ ║
║       │                                                                  │ ║
║       └── tiles ─► [VAEEncode] ─► [KSampler] ─► [VAEDecode]             │ ║
║                              rendered_tiles ↓                            │ ║
║                                                                          │ ║
║  [BBB_TileEqualizer]  ◄── rendered_tiles (sau KSampler)                  │ ║
║       ◄── original_tiles (từ Tiler, trước KSampler) ──────────────────── ┘ ║
║       ◄── strength                                                         ║
║       │                                                                    ║
║       ▼  equalized_tiles (same batch shape)                                ║
║                                                                            ║
║  [BBB_FluxStitcher] ◄── equalized_tiles                                    ║
║       ▼                                                                    ║
║  [Save]                                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CÁCH HOẠT ĐỘNG:                                                           ║
║                                                                            ║
║  Với mỗi tile i:                                                           ║
║    1. Lấy "original tile" tương ứng (chưa qua KSampler)                   ║
║    2. Tính màu mục tiêu (target) = màu original tile (upscaled, chưa ai)  ║
║    3. Transfer màu từ original → rendered bằng Reinhard Lab color transfer ║
║    4. Blend với strength (0=giữ nguyên rendered, 1=copy hoàn toàn màu gốc) ║
║                                                                            ║
║  Kết quả: mỗi rendered tile có màu khớp với original tile tương ứng       ║
║  → Không còn color drift giữa các tiles khi stitch                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETTINGS GỢI Ý:
  strength:     0.7 - 0.85  (cao quá → mất texture AI, thấp quá → drift còn)
  match_mode:   "Lab (Reinhard)" cho ảnh architectural (natural color)
                "RGB (mean+std)"  cho ảnh abstract / stylized (giữ vibe AI)
  scope:        "global"    → normalize toàn bộ tile về 1 target chung
                "per-tile"  → normalize từng tile độc lập về original tile đó
"""

import torch
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Lab color space (pure torch, không cần external lib)
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] → linear RGB (remove gamma)."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_rgb(linear: torch.Tensor) -> torch.Tensor:
    """Linear RGB → sRGB [0,1] (apply gamma)."""
    linear = linear.clamp(0.0, 1.0)
    return torch.where(linear <= 0.0031308,
                       linear * 12.92,
                       1.055 * linear ** (1.0 / 2.4) - 0.055)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB [B,C,H,W] → Lab [B,3,H,W].
    L: 0..100  |  a,b: roughly -128..127
    D65 illuminant, sRGB color space.
    """
    lin = _rgb_to_linear(rgb.clamp(0, 1))

    # RGB → XYZ (D65)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ → Lab (D65 reference white: Xn=0.95047, Yn=1.0, Zn=1.08883)
    def f(t):
        delta = 6.0 / 29.0
        return torch.where(t > delta**3,
                           t.clamp(min=1e-10) ** (1/3),
                           t / (3 * delta**2) + 4.0/29.0)

    fx = f(x / 0.95047)
    fy = f(y / 1.00000)
    fz = f(z / 1.08883)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)

    return torch.cat([L, a, b_ch], dim=1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Lab [B,3,H,W] → RGB [B,C,H,W] [0,1]."""
    L, a, b_ch = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_ch / 200.0

    def f_inv(t):
        delta = 6.0 / 29.0
        return torch.where(t > delta,
                           t ** 3,
                           3 * delta**2 * (t - 4.0/29.0))

    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.00000
    z = f_inv(fz) * 1.08883

    # XYZ → linear RGB
    r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

    linear = torch.cat([r, g, b], dim=1)
    return _linear_to_rgb(linear)


def _reinhard_transfer(src: torch.Tensor, ref: torch.Tensor,
                       mode: str = "Lab") -> torch.Tensor:
    """
    Reinhard color transfer: src → màu ref.

    src, ref: [B, C, H, W] trong RGB [0,1]
    mode: "Lab" hoặc "RGB"
    Returns: src với màu chuyển sang màu của ref.
    """
    if mode == "Lab":
        src_lab = _rgb_to_lab(src)
        ref_lab = _rgb_to_lab(ref)

        result_lab = torch.zeros_like(src_lab)
        for c in range(3):
            sm = src_lab[:, c].mean()
            ss = src_lab[:, c].std().clamp(min=1e-8)
            rm = ref_lab[:, c].mean()
            rs = ref_lab[:, c].std().clamp(min=1e-8)
            result_lab[:, c] = (src_lab[:, c] - sm) / ss * rs + rm

        return _lab_to_rgb(result_lab).clamp(0, 1)

    else:  # RGB mean+std
        result = torch.zeros_like(src)
        for c in range(src.shape[1]):
            sm = src[:, c].mean()
            ss = src[:, c].std().clamp(min=1e-8)
            rm = ref[:, c].mean()
            rs = ref[:, c].std().clamp(min=1e-8)
            result[:, c] = (src[:, c] - sm) / ss * rs + rm
        return result.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# NODE: BBB_TileEqualizer
# ─────────────────────────────────────────────────────────────────────────────

class BBB_TileEqualizer:
    """
    🎨 BBB Tile Equalizer
    ──────────────────────
    Normalize màu từng tile về cùng baseline trước khi stitch.

    Giải quyết: color drift / tone shift giữa các tiles sau KSampler.

    INPUT:
      rendered_tiles   → output của VAEDecode (sau KSampler), batch [N,H,W,C]
      original_tiles   → output của BBB_FluxTiler (trước KSampler), batch [N,H,W,C]
                         → dùng làm "target color reference" cho từng tile
      match_mode       → "Lab (Reinhard)" hoặc "RGB (mean+std)"
      scope            → "per-tile" hoặc "global"
      strength         → mức độ blend (0=giữ nguyên rendered, 1=copy màu gốc)

    OUTPUT:
      equalized_tiles  → batch [N,H,W,C], cùng shape với rendered_tiles
                         → kết nối vào BBB_FluxStitcher (thay cho rendered_tiles)

    LƯU Ý:
      - "per-tile": mỗi tile cân bằng màu độc lập → tốt nhất cho color consistency
      - "global": tất cả tiles normalize về average của toàn bộ original
                  → tốt khi original tiles quá khác nhau (vd: một tile toàn trời, một tile toàn tường)
    """

    MATCH_MODES = ["Lab (Reinhard)", "RGB (mean+std)"]
    SCOPES      = ["per-tile", "global"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rendered_tiles":  ("IMAGE",),
                "original_tiles":  ("IMAGE",),
                "match_mode": (cls.MATCH_MODES, {
                    "default": "Lab (Reinhard)",
                    "tooltip": (
                        "Lab (Reinhard): chính xác hơn, giữ màu tự nhiên — dùng cho arch-viz.\n"
                        "RGB (mean+std): nhanh hơn, phù hợp ảnh abstract/stylized."
                    ),
                }),
                "scope": (cls.SCOPES, {
                    "default": "per-tile",
                    "tooltip": (
                        "per-tile: normalize từng tile về original tile tương ứng (tốt nhất).\n"
                        "global: normalize tất cả tiles về average của toàn bộ original "
                        "(dùng khi có tile bị thiếu hoặc mismatch index)."
                    ),
                }),
                "strength": ("FLOAT", {
                    "default": 0.80,
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "0.0 = giữ nguyên rendered (không equalize)\n"
                        "1.0 = copy hoàn toàn màu sắc từ original\n"
                        "Khuyến nghị: 0.7 - 0.85 cho architectural viz."
                    ),
                }),
            },
            "optional": {
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "In per-tile color stats ra console để debug.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("equalized_tiles",)
    FUNCTION      = "equalize"
    CATEGORY      = "BBB/TilePro"

    def equalize(self, rendered_tiles, original_tiles, match_mode, scope,
                 strength, debug_log=False):
        N = rendered_tiles.shape[0]
        dev = rendered_tiles.device

        # Convert [N,H,W,C] → [N,C,H,W] để process
        rend = rendered_tiles.to(torch.float32).permute(0, 3, 1, 2)  # [N,C,H,W]
        orig = original_tiles.to(torch.float32).permute(0, 3, 1, 2)  # [N,C,H,W]
        orig = orig.to(dev)

        # Resize original nếu size khác (VAE có thể resize)
        if orig.shape[2:] != rend.shape[2:]:
            orig = F.interpolate(orig, size=rend.shape[2:], mode="bilinear", align_corners=False)

        # Trim N nếu không match (safety)
        n_ref = min(N, orig.shape[0])

        mode_key = "Lab" if "Lab" in match_mode else "RGB"

        if scope == "global":
            # Tính global reference từ average của tất cả original tiles
            global_ref = orig[:n_ref].mean(dim=0, keepdim=True)  # [1,C,H,W]
            global_ref = global_ref.expand(N, -1, -1, -1)        # [N,C,H,W]

        result = rend.clone()

        for i in range(N):
            src_tile = rend[i:i+1]   # [1,C,H,W]

            if scope == "per-tile" and i < n_ref:
                ref_tile = orig[i:i+1]
            elif scope == "global":
                ref_tile = global_ref[i:i+1]
            else:
                # i >= n_ref: không có original tile → skip
                continue

            transferred = _reinhard_transfer(src_tile, ref_tile, mode=mode_key)
            result[i:i+1] = src_tile * (1 - strength) + transferred * strength

            if debug_log:
                src_mean = src_tile.mean(dim=[2, 3]).squeeze().tolist()
                ref_mean = ref_tile.mean(dim=[2, 3]).squeeze().tolist()
                res_mean = result[i:i+1].mean(dim=[2, 3]).squeeze().tolist()
                print(
                    f"[BBB_TileEqualizer] Tile {i:3d}/{N}: "
                    f"src_rgb=({src_mean[0]:.3f},{src_mean[1]:.3f},{src_mean[2]:.3f}) "
                    f"ref_rgb=({ref_mean[0]:.3f},{ref_mean[1]:.3f},{ref_mean[2]:.3f}) "
                    f"out_rgb=({res_mean[0]:.3f},{res_mean[1]:.3f},{res_mean[2]:.3f})"
                )

        result = result.clamp(0.0, 1.0).permute(0, 2, 3, 1)  # [N,H,W,C]

        print(
            f"[BBB_TileEqualizer] {N} tiles equalized "
            f"| mode={match_mode} | scope={scope} | strength={strength}"
        )
        return (result,)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRATIONS
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BBB_TileEqualizer": BBB_TileEqualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBB_TileEqualizer": "🎨 BBB Tile Equalizer (normalize màu trước stitch)",
}
