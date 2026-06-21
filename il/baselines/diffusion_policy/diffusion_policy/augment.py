"""GPU-batched, label-preserving RGB augmentation for the image Diffusion Policy.

Plugged into ``train_rgbd.py``'s ``Agent.encode_obs`` via the existing ``self.aug`` hook.
It is applied ONLY during training (``eval_mode=False``) and only to the RGB channels.

Why these augmentations are safe
--------------------------------
Color jitter (brightness / contrast / saturation / hue), additive Gaussian noise and small
random translations all change *appearance* without changing the *meaning* of the scene, so
the demonstrated action stays a valid target. They effectively multiply the dataset and let
you train for more iterations without overfitting the exact training pixels.

Why there is NO horizontal flip here
------------------------------------
A horizontal mirror changes the geometry of the scene. To keep the data correct you would
also have to mirror (a) the robot proprioception in ``obs["state"]`` (joint angles / TCP pose
- not a simple sign flip for a 7-DoF arm) and (b) the action labels (the end-effector
delta-xyz, mapped through the camera->world frame). Flipping the image alone makes the image
inconsistent with the state and the action targets, which corrupts training. It is left out on
purpose; add it only if you also transform the state and actions accordingly.

Everything is implemented with plain tensor ops so the module holds NO parameters or buffers
- a checkpoint trained with augmentation loads identically into an eval Agent built without it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rec.601 luma weights, used for the contrast/saturation gray reference.
_LUMA = (0.299, 0.587, 0.114)


def _rand(b, device, lo, hi):
    """Per-sample factor in [lo, hi] with broadcastable shape (B, 1, 1, 1)."""
    return torch.empty(b, 1, 1, 1, device=device).uniform_(lo, hi)


def _rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    v = maxc
    delta = maxc - minc
    safe_delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    s = torch.where(maxc > 0, delta / torch.where(maxc == 0, torch.ones_like(maxc), maxc),
                    torch.zeros_like(maxc))
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    return h, s, v


def _pick(i, c0, c1, c2, c3, c4, c5):
    out = torch.where(i == 0, c0, c1)
    out = torch.where(i == 2, c2, out)
    out = torch.where(i == 3, c3, out)
    out = torch.where(i == 4, c4, out)
    out = torch.where(i == 5, c5, out)
    return out


class RGBAug(nn.Module):
    """Batched, per-sample RGB augmentation.

    Args (all strengths are the half-width of a uniform sampling range):
        brightness / contrast / saturation: factor ~ U[1-s, 1+s]
        hue: shift in hue fraction ~ U[-s, s]  (s in [0, 0.5])
        noise: std of additive Gaussian noise (in [0, 1] pixel units)
        translate: max random shift in pixels (reflection-padded crop); 0 disables
        p: probability of applying the *whole* augmentation to each sample
    """

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05,
                 noise=0.0, translate=0, p=1.0):
        super().__init__()
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.noise = float(noise)
        self.translate = int(translate)
        self.p = float(p)

    def _color_jitter(self, x):
        B = x.shape[0]
        dev = x.device
        w = torch.tensor(_LUMA, device=dev, dtype=x.dtype).view(1, 3, 1, 1)

        if self.brightness > 0:
            x = x * _rand(B, dev, 1 - self.brightness, 1 + self.brightness)
            x = x.clamp(0.0, 1.0)
        if self.contrast > 0:
            mean = (x * w).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * _rand(B, dev, 1 - self.contrast, 1 + self.contrast) + mean
            x = x.clamp(0.0, 1.0)
        if self.saturation > 0:
            gray = (x * w).sum(dim=1, keepdim=True)
            x = gray + (x - gray) * _rand(B, dev, 1 - self.saturation, 1 + self.saturation)
            x = x.clamp(0.0, 1.0)
        if self.hue > 0:
            h, s, v = _rgb_to_hsv(x.clamp(0.0, 1.0))
            dh = _rand(B, dev, -self.hue, self.hue).view(B, 1, 1)
            h = (h + dh) % 1.0
            x = _hsv_from(h, s, v)
        return x

    def _translate(self, x):
        if self.translate <= 0:
            return x
        pad = self.translate
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        B, _, H, W = x.shape
        dev = x.device
        offy = torch.randint(0, 2 * pad + 1, (B,), device=dev)
        offx = torch.randint(0, 2 * pad + 1, (B,), device=dev)
        # gather per-sample crops via grid (vectorised)
        ys = (torch.arange(H, device=dev).view(1, H, 1) + offy.view(B, 1, 1))
        xs = (torch.arange(W, device=dev).view(1, 1, W) + offx.view(B, 1, 1))
        ys = ys.unsqueeze(1).expand(B, x.shape[1], H, W)
        xs = xs.unsqueeze(1).expand(B, x.shape[1], H, W)
        bidx = torch.arange(B, device=dev).view(B, 1, 1, 1).expand(B, x.shape[1], H, W)
        cidx = torch.arange(x.shape[1], device=dev).view(1, -1, 1, 1).expand(B, x.shape[1], H, W)
        return xp[bidx, cidx, ys, xs]

    def forward(self, img):
        """img: (N, C, H, W) float in [0, 1]. Color ops require C >= 3 (uses first 3 chans)."""
        if not self.training:
            return img
        out = img
        if out.shape[1] >= 3:
            rgb = out[:, :3]
            aug = self._color_jitter(rgb)
            if out.shape[1] == 3:
                out = aug
            else:
                out = torch.cat([aug, out[:, 3:]], dim=1)
        out = self._translate(out)
        if self.noise > 0:
            out = out + torch.randn_like(out) * self.noise
        out = out.clamp(0.0, 1.0)

        if self.p < 1.0:
            B = img.shape[0]
            keep = (torch.rand(B, 1, 1, 1, device=img.device) < self.p).to(img.dtype)
            out = keep * out + (1.0 - keep) * img
        return out


def _hsv_from(h, s, v):
    """h, s, v each (B, H, W) -> (B, 3, H, W) rgb in [0, 1]."""
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    i = i.long() % 6
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = _pick(i, v, q, p, p, t, v)
    g = _pick(i, t, v, v, q, p, p)
    b = _pick(i, p, p, t, v, v, q)
    return torch.stack([r, g, b], dim=1).clamp(0.0, 1.0)
