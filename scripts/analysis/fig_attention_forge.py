"""
fig_attention_maps.py — ViT attention over patch tokens, FOG vs non-FOG.

Loads FORGE mae_2d_ep7 backbone, monkey-patches the last MultiHeadAttention
to capture attention weights, then overlays them on the CWT spectrogram.

Layout (3-panel):
  A  FOG window — CWT spectrogram
  B  FOG window — attention heatmap overlay (last ViT layer, mean head)
  C  Non-FOG window — attention heatmap overlay

Run: uv run python research/paper_mae/scripts_new/fig_attention_maps.py
"""

from pathlib import Path
import sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, zoom

ROOT      = Path(__file__).resolve().parents[2]   # repository root
sys.path.insert(0, str(ROOT))

CKPT_PATH = ROOT / "checkpoints/ssl_comparison/mae_2d_ep7.ckpt"
OUT       = ROOT / "research/paper_final/figures"
OUT.mkdir(parents=True, exist_ok=True)

FS   = 100
CWT_FMIN, CWT_FMAX = 0.3, 50.0     # true CWT span (WaveletTransform, 100 scales)
FMIN, FMAX = 0.5, 12.0             # displayed log-freq range — matches Fig 6

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        8,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

FOG_BG    = "#deebf7"; FOG_TXT    = "#08306b"
NONFOG_BG = "#deebf7"; NONFOG_TXT = "#2171b5"


# ── Model loading ─────────────────────────────────────────────────────────────

_captured_attn = {}
_hooks         = []


def load_backbone():
    import hydra
    ck  = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    cfg = ck["hyper_parameters"]["config"]
    sd  = ck["state_dict"]

    from utils.paths import normalize_data_paths
    normalize_data_paths(cfg.data.paths)

    backbone  = hydra.utils.instantiate(cfg.model.backbone.model_dump())
    transform = hydra.utils.instantiate(cfg.model.transform.model_dump())

    bk_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    tr_sd = {k[len("transform."):]: v for k, v in sd.items() if k.startswith("transform.")}
    backbone.load_state_dict(bk_sd, strict=False)
    transform.load_state_dict(tr_sd, strict=False)
    backbone.eval()
    transform.eval()

    # Hook the last TransformerEncoderLayer's self_attn (nn.MultiheadAttention)
    # SpectralPatchEncoder uses backbone.transformer.layers[-1].self_attn
    last_self_attn = backbone.transformer.layers[-1].self_attn

    # Pre-hook: force need_weights=True so PyTorch returns attention maps
    def pre_hook(module, args, kwargs):
        kwargs["need_weights"]        = True
        kwargs["average_attn_weights"] = False   # per-head [B, H, N, N]
        return args, kwargs

    # Post-hook: capture the returned attention weights
    def post_hook(module, inputs, output):
        # output = (attn_output [B,N,D], attn_weights [B,H,N,N] or None)
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            _captured_attn["weights"] = output[1].detach().cpu()

    _hooks.append(last_self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
    _hooks.append(last_self_attn.register_forward_hook(post_hook))
    print(f"Hooked transformer.layers[-1].self_attn")

    freq_patch = cfg.model.backbone.model_dump().get("freq_patch", 20)
    time_patch = cfg.model.backbone.model_dump().get("time_patch", 50)
    return backbone, transform, freq_patch, time_patch


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zarr_and_masks():
    import zarr
    z    = zarr.open(str(ROOT / "data/processed/len1000_stride200_kaggle.zarr"))
    cl   = z["metadata/class_label"][:]
    prot = np.array(z["metadata/protocol"][:])
    pur  = z["metadata/purity"][:]
    pid  = np.array(z["metadata/patient_id"][:])
    fog_mask    = (cl == 1) & (prot == "tdcsfog") & (pur >= 0.8)
    nonfog_mask = (cl == 0) & (prot == "tdcsfog") & (pur >= 0.8)
    return z, fog_mask, nonfog_mask, pid


def select_fog_window(z, fog_mask, pid, seed=42):
    rng   = np.random.default_rng(seed)
    p_sel = sorted(
        [(int(np.sum(fog_mask & (pid == p))), p) for p in np.unique(pid[fog_mask])],
        reverse=True
    )[0][1]
    idx = int(rng.choice(np.where(fog_mask & (pid == p_sel))[0]))
    print(f"FOG window: idx={idx}, patient={p_sel}")
    return np.array(z["accs"][idx]).astype(float), p_sel


@torch.no_grad()
def select_nonfog_window(backbone, transform, z, nonfog_mask,
                         freq_patch, n_candidates=40, seed=0):
    """Search across all non-FOG candidates; return the window with the
    lowest freeze-band attention energy AND sufficient signal variance."""
    rng    = np.random.default_rng(seed)
    all_idx = np.where(nonfog_mask)[0]
    sample  = rng.choice(all_idx, size=min(n_candidates, len(all_idx)), replace=False)

    # Which freq-patch rows cover 3-8 Hz? (use the model's TRUE freq axis)
    freqs       = model_freq_axis(transform)
    n_fp        = 100 // freq_patch
    freeze_rows = [r for r in range(n_fp)
                   if freqs[min((r+1)*freq_patch-1, 99)] >= 3.0
                   and freqs[r*freq_patch] <= 8.0]
    freeze_rows = freeze_rows or [2, 3]

    # Minimum signal variance threshold (exclude near-silent windows)
    var_threshold = np.percentile(
        [float(np.array(z["accs"][i]).std()) for i in sample[:10]], 25
    )

    best_idx, best_score = int(sample[0]), float("inf")
    for idx in sample:
        win = np.array(z["accs"][idx]).astype(float)
        if win.std() < var_threshold:
            continue
        attn_map, _, _ = get_attention(backbone, transform, win, freq_patch, 50)
        attn_norm   = attn_map / (attn_map.sum() + 1e-8)
        score       = float(attn_norm[freeze_rows].sum())
        print(f"  non-FOG candidate {idx}: freeze={score:.3f}  var={win.std():.3f}")
        if score < best_score:
            best_score, best_idx = score, int(idx)

    print(f"Selected non-FOG idx={best_idx} (freeze-band score={best_score:.3f})")
    return np.array(z["accs"][best_idx]).astype(float)


# ── Inference + attention extraction ─────────────────────────────────────────

@torch.no_grad()
def get_attention(backbone, transform, win, freq_patch, time_patch):
    """Return attention map [n_freq_patches, n_time_patches] and CWT [F, T]."""
    x = torch.tensor(win.T, dtype=torch.float32).unsqueeze(0)   # [1, C, T]
    spec = transform(x)                                           # [1, C, F, T]
    _captured_attn.clear()
    _ = backbone(spec)                                            # populates _captured_attn
    attn = _captured_attn["weights"]                              # [1, H, N, N]

    # Mean over heads, mean over query positions → [N]
    attn_mean = attn[0].mean(dim=0).mean(dim=0).numpy()          # [N]

    # CWT for display (best channel)
    spec_np = spec[0].numpy()                                     # [C, F, T]
    # Display a single representative axis (the most dynamic one) of the exact spec
    # the model patched — NOT a channel average (the specs are per-channel
    # instance-normed, so averaging them is not meaningful). Attention aligns
    # because we plot it on the same model freq axis.
    best_ch = int(np.diff(win, axis=0).std(axis=0).argmax())     # most dynamic axis
    cwt_disp = spec_np[best_ch]                                   # [F, T]
    freqs = model_freq_axis(transform)                           # model's true non-uniform axis

    n_freq = cwt_disp.shape[0] // freq_patch
    n_time = cwt_disp.shape[1] // time_patch
    attn_map = attn_mean[:n_freq * n_time].reshape(n_freq, n_time)
    return attn_map, cwt_disp, freqs


# ── CWT frequency axis from transform ────────────────────────────────────────

def model_freq_axis(transform):
    """The transform's TRUE (FOG-optimized, non-uniform) freq axis, aligned with
    the spec rows / attention patch grid. freqs = fc*fs/scales (morl fc=0.8125)."""
    return 0.8125 * float(transform.fs) / transform.scales.detach().cpu().numpy()


# ── Plotting ─────────────────────────────────────────────────────────────────

def _setup_log_freq_ax(ax, freqs, T, show_ylabel):
    ax.set_yscale("log")
    ax.set_ylim(FMIN, FMAX)                  # display 0.5-12 Hz (matches Fig 6)
    ax.set_xlim(0, T / FS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"
    ))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    if show_ylabel:
        ax.set_ylabel("Freq (Hz)", fontsize=7)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel("Time (s)", fontsize=7)


def plot_spectrogram(ax, cwt, freqs, title, bg, fg, show_ylabel=True):
    from scipy.ndimage import gaussian_filter
    # cwt = the model spec (already log1p power + instance-normed); light smoothing
    # only. Contrast scaled on the DISPLAYED band (0.5-12 Hz) so the bands aren't
    # washed out; high vmin floor darkens the background toward black (Fig-6 look).
    sm = gaussian_filter(cwt.astype(np.float64), sigma=(1.5, 3.0))
    fmask = (freqs >= FMIN) & (freqs <= FMAX)
    vmin, vmax = float(np.percentile(sm[fmask], 35)), float(np.percentile(sm[fmask], 99.5))
    t = np.linspace(0, cwt.shape[1] / FS, cwt.shape[1])
    ax.pcolormesh(t, freqs, sm, cmap="Blues", vmin=vmin, vmax=vmax,
                  shading="auto", rasterized=True)
    ax.set_title(title, fontsize=8, fontweight="bold", color=fg,
                 bbox=dict(facecolor=bg, edgecolor="none", pad=2))
    _setup_log_freq_ax(ax, freqs, cwt.shape[1], show_ylabel)


def plot_attention(ax, attn_map, freqs, T, freq_patch, time_patch,
                   title, bg, fg, show_ylabel=True):
    """Standalone attention heatmap — same log-freq / time axes as CWT."""
    F = len(freqs)
    nfp, ntp = attn_map.shape
    zoom_f = F / (nfp * freq_patch)
    zoom_t = T / (ntp * time_patch)
    attn_up = zoom(
        attn_map.repeat(freq_patch, axis=0).repeat(time_patch, axis=1),
        (zoom_f, zoom_t), order=1
    )
    attn_sm   = gaussian_filter(attn_up, sigma=(2.0, 3.0))
    attn_norm = (attn_sm - attn_sm.min()) / (attn_sm.max() - attn_sm.min() + 1e-8)

    t = np.linspace(0, T / FS, T)
    ax.pcolormesh(t, freqs, attn_norm, cmap="Blues", vmin=0, vmax=1,
                  shading="auto", rasterized=True)

    # Mark the 3–8 Hz freeze band (reference annotation lines → red dashed)
    BAND_CLR = "#d62728"
    ax.axhline(3, color=BAND_CLR, lw=0.9, ls="--", alpha=0.90, zorder=5)
    ax.axhline(8, color=BAND_CLR, lw=0.9, ls="--", alpha=0.90, zorder=5)
    ax.text(0.97, 0.42, "3–8 Hz\nfreeze band",
            transform=ax.transAxes, fontsize=6, color=BAND_CLR,
            ha="right", va="center", zorder=6)

    ax.set_title(title, fontsize=8, fontweight="bold", color=fg,
                 bbox=dict(facecolor=bg, edgecolor="none", pad=2))
    _setup_log_freq_ax(ax, freqs, T, show_ylabel)


def main():
    print("Loading backbone...")
    backbone, transform, freq_patch, time_patch = load_backbone()

    print("Loading data...")
    z, fog_mask, nonfog_mask, pid = load_zarr_and_masks()
    fog_win, p_sel = select_fog_window(z, fog_mask, pid, seed=42)

    print("Extracting attention (FOG window)...")
    attn_fog, cwt_fog, freqs_fog = get_attention(backbone, transform, fog_win, freq_patch, time_patch)

    print("Searching for non-FOG window with low freeze-band attention...")
    nonfog_win = select_nonfog_window(backbone, transform, z, nonfog_mask,
                                      freq_patch, n_candidates=40, seed=0)
    print("Extracting attention (non-FOG window)...")
    attn_nonfog, cwt_nonfog, freqs_nonfog = get_attention(backbone, transform, nonfog_win, freq_patch, time_patch)

    print(f"  attn_fog shape: {attn_fog.shape}  cwt_fog: {cwt_fog.shape}")

    # ── Figure: 2 rows × 2 cols (CWT | standalone attention heatmap) ────────
    # wide-and-flat panels to match Fig 6's spectrogram aspect ratio
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.14,
                                          "left": 0.05, "right": 0.99,
                                          "top": 0.92, "bottom": 0.10})

    T_fog    = cwt_fog.shape[1]
    T_nonfog = cwt_nonfog.shape[1]

    # Row 0: FOG
    plot_spectrogram(axes[0, 0], cwt_fog,    freqs_fog,    "FOG — CWT",
                     FOG_BG, FOG_TXT)
    plot_attention(axes[0, 1], attn_fog, freqs_fog, T_fog, freq_patch, time_patch,
                   "FOG — Encoder Attention", FOG_BG, FOG_TXT, show_ylabel=False)

    # Row 1: Non-FOG
    plot_spectrogram(axes[1, 0], cwt_nonfog, freqs_nonfog, "Non-FOG — CWT",
                     NONFOG_BG, NONFOG_TXT)
    plot_attention(axes[1, 1], attn_nonfog, freqs_nonfog, T_nonfog, freq_patch, time_patch,
                   "Non-FOG — Encoder Attention", NONFOG_BG, NONFOG_TXT, show_ylabel=False)

    fig.suptitle("FORGE encoder attention: last ViT layer, mean over heads",
                 fontsize=9, y=0.98)

    out_path = OUT / "figS3_attention_maps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
