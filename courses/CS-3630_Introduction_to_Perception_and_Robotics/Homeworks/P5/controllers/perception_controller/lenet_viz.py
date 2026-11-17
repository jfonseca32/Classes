"""
Pedagogy-only LeNet visualization for Parts 3 and 4.

Inspects whatever network the student defined in networks.LeNet: draws an
architecture/shape summary from a dummy forward pass, and an "X-ray" of
feature maps for a real input. Not graded.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import animation
from matplotlib.patches import Rectangle


def _as_bchw(image) -> torch.Tensor:
    """Accept (H,W), (1,H,W), or (1,1,H,W) and return a float batch tensor."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if not torch.is_tensor(image):
        raise TypeError(f"Expected tensor or ndarray, got {type(image)}")
    x = image.detach().float().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected 2D/3D/4D image, got shape {tuple(x.shape)}")
    return x


def _is_spatial(module: nn.Module) -> bool:
    return isinstance(
        module,
        (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d),
    )


def _collect_named_leaves(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Named modules that actually run compute (skip containers / activations shared oddly)."""
    leaves = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # Skip parent containers that have children already listed
        if any(True for _ in module.children()):
            continue
        leaves.append((name, module))
    return leaves


def summarize_architecture(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 1, 32, 32),
    title: str = "Your network (shapes from a dummy forward pass)",
) -> None:
    """
    Draw a vertical architecture diagram with output shapes from *this* model.

    Hooks whatever leaf modules the student registered. If forward is missing
    or broken, prints a clear message instead of a stack trace wall.
    """
    model = model.eval()
    device = next(model.parameters(), torch.tensor(0.0)).device
    dummy = torch.zeros(input_shape, device=device)

    shapes: list[tuple[str, str, tuple]] = []
    hooks = []

    def make_hook(name: str, module: nn.Module):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                shapes.append((name, type(module).__name__, tuple(t.shape)))

        return _hook

    for name, module in _collect_named_leaves(model):
        hooks.append(module.register_forward_hook(make_hook(name, module)))

    try:
        with torch.no_grad():
            out = model(dummy)
    except NotImplementedError:
        print(
            "LeNet.forward is not implemented yet. "
            "Fill in networks.LeNet.forward to unlock the architecture diagram."
        )
        return
    except Exception as e:
        print(
            f"Forward pass failed while probing architecture ({type(e).__name__}: {e}).\n"
            "Check your layer definitions in LeNet.__init__ and the order in forward."
        )
        return
    finally:
        for h in hooks:
            h.remove()

    if not shapes:
        print(
            "No intermediate activations were captured. "
            "Make sure forward actually calls your conv/linear modules."
        )
        return

    rows = [("input", "Tensor", tuple(dummy.shape))] + shapes
    if torch.is_tensor(out):
        rows.append(("output", "Tensor", tuple(out.shape)))

    n = len(rows)
    fig_h = max(3.5, 0.55 * n + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)

    for i, (name, kind, shape) in enumerate(rows):
        y = n - i
        shape_str = " x ".join(str(s) for s in shape)
        label = f"{name}  ·  {kind}\n{shape_str}"
        ax.add_patch(
            plt.Rectangle(
                (1.5, y - 0.35),
                7.0,
                0.7,
                fill=True,
                facecolor="#f4f1ea" if i == 0 else "#e8eef5",
                edgecolor="#2c3e50",
                linewidth=1.2,
                zorder=2,
            )
        )
        ax.text(5.0, y, label, ha="center", va="center", fontsize=9, zorder=3)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(5.0, y - 0.55),
                xytext=(5.0, y - 0.85),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
                zorder=1,
            )

    plt.tight_layout()
    plt.show()


def xray(
    model: nn.Module,
    image,
    class_names: list[str] | None = None,
    max_maps: int = 16,
    title: str | None = None,
) -> None:
    """
    Layer X-ray: input image, spatial feature-map grids, and class scores.

    Discovers Conv2d / pooling modules dynamically so the figure updates when
    students change architecture or forward.
    """
    model = model.eval()
    x = _as_bchw(image)
    device = next(model.parameters(), torch.tensor(0.0)).device
    x = x.to(device)

    activations: list[tuple[str, torch.Tensor]] = []
    hooks = []

    def make_hook(name: str):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.ndim == 4:
                activations.append((name, t.detach().cpu()))

        return _hook

    for name, module in model.named_modules():
        if name and _is_spatial(module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            logits = model(x)
    except NotImplementedError:
        print(
            "LeNet.forward is not implemented yet. "
            "Fill in networks.LeNet.forward to unlock the layer X-ray."
        )
        return
    except Exception as e:
        print(
            f"Forward pass failed during X-ray ({type(e).__name__}: {e}).\n"
            "Check LeNet.__init__ / forward, and that the input is 32x32 grayscale."
        )
        return
    finally:
        for h in hooks:
            h.remove()

    if not torch.is_tensor(logits):
        print("Model output was not a tensor; cannot plot class scores.")
        return

    logits = logits.detach().cpu().reshape(-1)
    probs = F.softmax(logits.float(), dim=0).numpy()
    pred = int(probs.argmax())
    if class_names is None:
        class_names = [str(i) for i in range(len(probs))]

    # Layout: input | up to a few spatial stages | bar chart
    stages = activations[:6]  # keep the figure readable
    n_panels = 1 + len(stages) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
    if n_panels == 1:
        axes = [axes]

    # Input
    img = x[0, 0].detach().cpu().numpy()
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=max(1.0, float(img.max())))
    axes[0].set_title("input", fontsize=10)
    axes[0].axis("off")

    for ax, (name, feat) in zip(axes[1:-1], stages, strict=False):
        maps = feat[0]  # (C,H,W)
        c = min(maps.shape[0], max_maps)
        grid = _feature_grid(maps[:c].numpy())
        ax.imshow(grid, cmap="viridis")
        ax.set_title(f"{name}\n{tuple(maps.shape)}", fontsize=9)
        ax.axis("off")

    # Class scores
    ax = axes[-1]
    colors = ["#c0392b" if i == pred else "#7f8c8d" for i in range(len(probs))]
    ax.barh(range(len(probs)), probs, color=colors)
    ax.set_yticks(range(len(probs)))
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("softmax")
    ax.set_title(f"pred = {class_names[pred]}", fontsize=10)

    fig.suptitle(
        title or "Layer X-ray (your LeNet)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def _feature_grid(maps: np.ndarray, pad: int = 1) -> np.ndarray:
    """Tile (C,H,W) feature maps into a single 2D image for imshow."""
    c, h, w = maps.shape
    cols = int(np.ceil(np.sqrt(c)))
    rows = int(np.ceil(c / cols))
    # Per-map normalize for visibility
    tiles = []
    for i in range(c):
        m = maps[i]
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
        tiles.append(m)

    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad))
    for i, m in enumerate(tiles):
        r, col = divmod(i, cols)
        y = r * (h + pad)
        x = col * (w + pad)
        canvas[y : y + h, x : x + w] = m
    return canvas


def _capture_forward(
    model: nn.Module, image
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]], np.ndarray] | None:
    """Run one forward pass; return (input_bchw_cpu, spatial_activations, probs) or None."""
    model = model.eval()
    x = _as_bchw(image)
    device = next(model.parameters(), torch.tensor(0.0)).device
    x = x.to(device)

    activations: list[tuple[str, torch.Tensor]] = []
    hooks = []

    def make_hook(name: str):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.ndim == 4:
                activations.append((name, t.detach().cpu()))

        return _hook

    for name, module in model.named_modules():
        if name and _is_spatial(module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            logits = model(x)
    except NotImplementedError:
        print(
            "LeNet.forward is not implemented yet. "
            "Fill in networks.LeNet.forward to unlock the animations."
        )
        return None
    except Exception as e:
        print(
            f"Forward pass failed ({type(e).__name__}: {e}).\n"
            "Check LeNet.__init__ / forward, and that the input is 32x32 grayscale."
        )
        return None
    finally:
        for h in hooks:
            h.remove()

    if not torch.is_tensor(logits):
        print("Model output was not a tensor; cannot animate class scores.")
        return None

    probs = F.softmax(logits.detach().cpu().reshape(-1).float(), dim=0).numpy()
    return x.detach().cpu(), activations, probs


def _display_or_save(
    anim: animation.FuncAnimation,
    save_path: str | Path | None,
    fps: int = 8,
):
    """Save a GIF if asked; otherwise try notebook HTML display, else plt.show()."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(path), writer=animation.PillowWriter(fps=fps))
        print(f"saved animation -> {path}")
        plt.close(anim._fig)
        return anim

    try:
        from IPython.display import HTML, display

        plt.close(anim._fig)
        display(HTML(anim.to_jshtml()))
    except Exception:
        plt.show()
    return anim


def _normalize_map(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi > lo:
        return (m - lo) / (hi - lo)
    return np.zeros_like(m, dtype=float)


def _downsample_map(m: np.ndarray, max_side: int = 14) -> np.ndarray:
    """Nearest-neighbor downsample so 3D surfaces stay cheap to render."""
    h, w = m.shape
    if max(h, w) <= max_side:
        return m
    ys = np.linspace(0, h - 1, max_side).astype(int)
    xs = np.linspace(0, w - 1, max_side).astype(int)
    return m[ys][:, xs]


def _draw_image_plane(
    ax,
    img2d: np.ndarray,
    depth: float,
    cmap,
    size: float = 2.4,
    x_offset: float = 0.0,
    z_offset: float = 0.0,
    alpha: float = 0.95,
):
    """Draw a feature map as a textured rectangle in 3D (depth along Y)."""
    img2d = _normalize_map(img2d.astype(float))
    h, w = img2d.shape
    # Plane spans X (width) and Z (height); Y is network depth
    xs = np.linspace(-size / 2, size / 2, w) + x_offset
    zs = np.linspace(-size / 2, size / 2, h) + z_offset
    X, Z = np.meshgrid(xs, zs)
    Y = np.full_like(X, depth)
    colors = cmap(img2d)
    colors[..., -1] = alpha
    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=colors,
        shade=False,
        linewidth=0,
        antialiased=False,
    )


def animate_forward_pass(
    model: nn.Module,
    image,
    class_names: list[str] | None = None,
    max_maps: int = 4,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    3D forward-pass showcase: feature-map planes stacked in depth, revealed
    layer by layer with a slow camera orbit, then softmax bars rise at the end.

    Layer names live in a fixed 2D sidebar so they stay readable while the
    camera moves.
    """
    from matplotlib import cm as mpl_cm

    captured = _capture_forward(model, image)
    if captured is None:
        return None
    x, activations, probs = captured

    stages = [
        (name, feat) for name, feat in activations if feat.shape[-2] > 1 and feat.shape[-1] > 1
    ][:4]
    pred = int(probs.argmax())
    if class_names is None:
        class_names = [str(i) for i in range(len(probs))]

    # (label, display maps, cmap, n_channels_total, native HxW)
    layers: list[tuple[str, list[np.ndarray], str, int, tuple[int, int]]] = []
    inp_full = x[0, 0].numpy()
    layers.append(("input", [_downsample_map(inp_full)], "gray", 1, tuple(inp_full.shape)))
    for name, feat in stages:
        maps = feat[0].numpy()
        n_show = min(maps.shape[0], max_maps)
        layer_maps = [_downsample_map(maps[i]) for i in range(n_show)]
        layers.append(
            (name, layer_maps, "viridis", int(maps.shape[0]), (maps.shape[1], maps.shape[2]))
        )

    # Large depth gaps so perspective still separates layers
    depth_pitch = 7.5
    channel_dx = 0.10
    channel_dz = 0.035
    bar_depth = (len(layers) - 1) * depth_pitch + 5.5

    fig = plt.figure(figsize=(12.5, 6.8))
    ax = fig.add_axes([0.30, 0.06, 0.68, 0.84], projection="3d")
    fig.suptitle(
        title or "3D forward pass through your LeNet",
        fontsize=15,
        y=0.96,
        fontweight="bold",
    )

    reveal_hold = 14
    n_layers = len(layers)
    score_steps = 18
    end_hold = 28
    total = n_layers * reveal_hold + score_steps + end_hold
    base_azim = -52
    base_elev = 18
    azim_speed = 0.30  # slow orbit

    legend = fig.text(
        0.025,
        0.78,
        "",
        ha="left",
        va="top",
        fontsize=11,
        family="DejaVu Sans Mono",
        linespacing=1.7,
        color="#1a1a1a",
    )
    status = fig.text(
        0.025,
        0.10,
        "",
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="#2c3e50",
    )

    def _cmap(name: str):
        try:
            return plt.colormaps[name]
        except (AttributeError, KeyError):
            return mpl_cm.get_cmap(name)

    def _legend_text(visible_layers: int) -> str:
        lines = ["Layers", "------"]
        for i, (name, maps, _cmap_name, n_total, hw) in enumerate(layers):
            marker = ">" if i == visible_layers - 1 else " "
            dim = f"{hw[0]}x{hw[1]}"
            shown = f"show {len(maps)}/{n_total}" if n_total != len(maps) else f"{n_total} maps"
            if i < visible_layers:
                lines.append(f"{marker} {name}")
                lines.append(f"    {shown} · {dim}")
            else:
                lines.append(f"  {name}")
                lines.append("    (not yet)")
        return "\n".join(lines)

    def draw_scene(visible_layers: int, score_frac: float, azim: float, elev: float):
        ax.clear()
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-1.0, bar_depth + 1.5)
        ax.set_zlim(-2.2, 2.2)
        # Stretch depth (Y) so layer gaps don't collapse in perspective
        with contextlib.suppress(Exception):
            ax.set_box_aspect((1.0, 2.8, 1.0))

        for li in range(min(visible_layers, n_layers)):
            _name, maps, cmap_name, _n_total, hw = layers[li]
            depth = li * depth_pitch
            cmap = _cmap(cmap_name)
            n_maps = len(maps)
            # Size scales gently with native resolution
            size = 2.0 * (0.55 + 0.45 * (hw[0] / 32.0))
            for ci, m in enumerate(maps):
                mid = (n_maps - 1) / 2.0
                x_off = (ci - mid) * channel_dx
                z_off = (ci - mid) * channel_dz
                alpha = 0.97 if li == visible_layers - 1 else 0.62
                _draw_image_plane(
                    ax,
                    m,
                    depth=depth,
                    cmap=cmap,
                    size=size,
                    x_offset=x_off,
                    z_offset=z_off,
                    alpha=alpha,
                )

        if score_frac > 0:
            xs = np.linspace(-1.6, 1.6, len(probs))
            for i, (px, p) in enumerate(zip(xs, probs, strict=False)):
                h = float(p) * 1.8 * score_frac
                color = "#c0392b" if i == pred else "#b0b7bd"
                ax.bar3d(
                    px - 0.09,
                    bar_depth,
                    -2.0,
                    0.18,
                    0.50,
                    max(h, 0.03),
                    color=color,
                    shade=True,
                    alpha=0.92,
                )

    def update(frame: int):
        azim = base_azim + frame * azim_speed
        elev = base_elev

        if frame < n_layers * reveal_hold:
            visible = frame // reveal_hold + 1
            score_frac = 0.0
            name = layers[visible - 1][0]
            status.set_text(f"Step {visible}/{n_layers}\nNow showing: {name}")
        else:
            visible = n_layers
            t = frame - n_layers * reveal_hold
            if t < score_steps:
                score_frac = (t + 1) / float(score_steps)
                status.set_text("Class scores\n(softmax)")
            else:
                score_frac = 1.0
                status.set_text(f"Prediction: {class_names[pred]}")

        legend.set_text(_legend_text(visible))
        draw_scene(visible, score_frac, azim, elev)
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, interval=120, blit=False, repeat=True)
    return _display_or_save(anim, save_path, fps=8)


def animate_conv_kernel(
    model: nn.Module,
    image,
    layer_name: str = "conv1",
    out_channel: int = 0,
    step: int = 2,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Animate a 5x5 (or whatever) kernel from the student's Conv2d sliding over the
    input while the corresponding output feature map fills in.
    """
    model = model.eval()
    x = _as_bchw(image)
    img = x[0, 0].detach().cpu().numpy()

    conv = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            conv = module
            break
    if conv is None:
        # Fall back to the first Conv2d
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_name, conv = name, module
                break
    if conv is None:
        print("No Conv2d layer found on this model.")
        return None

    if conv.in_channels != 1:
        print(
            f"{layer_name} has in_channels={conv.in_channels}; "
            "kernel scan demo currently expects a single-channel input (conv1)."
        )
        return None

    weight = conv.weight.detach().cpu()  # (out, in, kH, kW)
    bias = conv.bias.detach().cpu() if conv.bias is not None else None
    out_channel = int(out_channel) % weight.shape[0]
    kernel = weight[out_channel, 0].numpy()
    k_h, k_w = kernel.shape
    stride_h, stride_w = (
        conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
    )
    # Only support zero padding for a clear sliding-window story
    pad = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    if pad != (0, 0):
        print(f"Note: {layer_name} padding={pad}; scan still uses valid convolution.")

    # Full output map for this channel via conv2d
    with torch.no_grad():
        out = F.conv2d(
            x[:, :1].cpu(),
            weight[out_channel : out_channel + 1],
            bias[out_channel : out_channel + 1] if bias is not None else None,
            stride=conv.stride,
            padding=conv.padding,
        )[0, 0].numpy()

    h_in, w_in = img.shape
    h_out, w_out = out.shape

    positions = [
        (oy, ox) for oy in range(0, h_out, max(1, step)) for ox in range(0, w_out, max(1, step))
    ]
    if positions[-1] != (h_out - 1, w_out - 1):
        positions.append((h_out - 1, w_out - 1))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    ax_in, ax_k, ax_out = axes

    ax_in.imshow(img, cmap="gray", vmin=0, vmax=max(1.0, float(img.max())))
    rect = Rectangle(
        (0, 0),
        k_w,
        k_h,
        fill=False,
        edgecolor="#f1c40f",
        linewidth=2.0,
    )
    ax_in.add_patch(rect)
    ax_in.set_title("input + kernel window", fontsize=10)
    ax_in.axis("off")

    k_show = kernel.copy()
    lo, hi = float(k_show.min()), float(k_show.max())
    k_norm = (k_show - lo) / (hi - lo) if hi > lo else np.zeros_like(k_show)
    ax_k.imshow(k_norm, cmap="coolwarm", vmin=0, vmax=1)
    ax_k.set_title(f"{layer_name} filter [{out_channel}]", fontsize=10)
    ax_k.set_xticks(range(k_w))
    ax_k.set_yticks(range(k_h))

    partial = np.full_like(out, np.nan)
    out_im = ax_out.imshow(
        np.zeros_like(out),
        cmap="viridis",
        vmin=float(out.min()),
        vmax=float(out.max()) if out.max() > out.min() else 1.0,
    )
    ax_out.set_title("output feature map", fontsize=10)
    ax_out.axis("off")

    fig.suptitle(
        title or f"Convolution scan · {layer_name} · channel {out_channel}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    # Hold a few frames at the end
    hold = 10
    frames = list(range(len(positions))) + [len(positions) - 1] * hold

    def update(frame_idx: int):
        oy, ox = positions[frame_idx]
        # Map output (oy,ox) back to input top-left of the window
        y0 = oy * stride_h
        x0 = ox * stride_w
        rect.set_xy((x0 - 0.5, y0 - 0.5))

        # Fill all outputs up to this raster position
        partial[:, :] = np.nan
        partial[:oy, :] = out[:oy, :]
        partial[oy, : ox + 1] = out[oy, : ox + 1]
        data = np.ma.array(partial, mask=np.isnan(partial))
        out_im.set_data(data)
        ax_out.set_title(f"output  ({oy},{ox})", fontsize=10)
        return rect, out_im

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False, repeat=True)
    return _display_or_save(anim, save_path, fps=16)


def preview_from_mnist(
    model: nn.Module,
    digit: int | None = None,
    dataset_root: str = "dataset/",
    show_architecture: bool = True,
) -> None:
    """Convenience for notebook cells: load one MNIST test image and X-ray it."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    test_set = datasets.MNIST(root=dataset_root, train=False, transform=transform, download=True)

    if digit is None:
        idx = 0
        image, label = test_set[idx]
    else:
        image, label = next((img, y) for img, y in test_set if int(y) == int(digit))

    if show_architecture:
        summarize_architecture(model)

    xray(
        model,
        image,
        title=f"Layer X-ray · MNIST digit {int(label)}",
    )
