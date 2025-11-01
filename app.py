\
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

# =========================
# Classic motif generators
# =========================
MOTIFS = {
    "diamond": np.array([[0,0,0,1,0,0,0,0],
                         [0,0,1,1,1,0,0,0],
                         [0,1,1,1,1,1,0,0],
                         [1,1,1,1,1,1,1,0],
                         [0,1,1,1,1,1,0,0],
                         [0,0,1,1,1,0,0,0],
                         [0,0,0,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0]], dtype=int),
    "heart":   np.array([[0,1,1,0,0,1,1,0],
                         [1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1],
                         [0,1,1,1,1,1,1,0],
                         [0,0,1,1,1,1,0,0],
                         [0,0,0,1,1,0,0,0],
                         [0,0,0,0,0,0,0,0]], dtype=int),
    "wave":    np.array([[0,0,1,1,0,0,1,1],
                         [0,1,1,0,0,1,1,0],
                         [1,1,0,0,1,1,0,0],
                         [1,0,0,1,1,0,0,1],
                         [1,0,0,1,1,0,0,1],
                         [1,1,0,0,1,1,0,0],
                         [0,1,1,0,0,1,1,0],
                         [0,0,1,1,0,0,1,1]], dtype=int),
    # Nordic-style template snowflake (tile symmetry)
    "nordic_snow": np.array([[0,0,1,0,0,1,0,0],
                             [0,1,1,1,1,1,1,0],
                             [1,1,1,1,1,1,1,1],
                             [0,1,1,1,1,1,1,0],
                             [0,0,1,1,1,1,0,0],
                             [0,1,1,0,0,1,1,0],
                             [0,0,1,0,0,1,0,0],
                             [0,0,0,0,0,0,0,0]], dtype=int),
}

def mirror_quad(seed):
    top = np.concatenate([seed, np.fliplr(seed)], axis=1)
    bottom = np.flipud(top)
    return np.concatenate([top, bottom], axis=0)

def upscale(grid, scale):
    return np.kron(grid, np.ones((scale, scale), dtype=int))

def generate_classic(motif="diamond", symmetry="quad", scale=2, mix_random=False, seed=0):
    base = MOTIFS[motif].copy()
    if mix_random:
        rng = np.random.default_rng(None if seed==0 else seed)
        noise = rng.integers(0, 2, size=base.shape, dtype=int)
        base = (base | noise).astype(int)
    if symmetry == "quad":
        mat = mirror_quad(base)
    elif symmetry == "horizontal":
        mat = np.concatenate([base, np.fliplr(base)], axis=1)
    elif symmetry == "vertical":
        mat = np.concatenate([base, np.flipud(base)], axis=0)
    else:
        mat = base
    if scale > 1:
        mat = upscale(mat, scale)
    return mat

# =========================
# D6 snowflake (true 6-fold)
# =========================
def _sectorize(theta):
    two_pi = 2*np.pi
    theta = (theta % two_pi)
    sector = np.floor(theta / (np.pi/6)).astype(int)
    return sector

def _reflect_to_base(theta):
    two_pi = 2*np.pi
    theta = (theta + np.pi) % two_pi - np.pi
    sector = _sectorize(theta)
    sector_center = (sector + 0.5)*(np.pi/6)
    theta_rel = theta - sector_center
    reflected = (sector % 2 == 1)
    theta_base = np.where(reflected, -theta_rel, theta_rel)
    return theta_base, reflected

def base_wedge_pattern(r, theta_base, params, rng):
    spikes = params.get("spikes", 5)
    sharpness = params.get("sharpness", 14.0)
    density = params.get("density", 0.35)
    ring_strength = params.get("ring_strength", 0.55)
    base_width = 18 * np.pi/180
    width = base_width * (1.0 - 0.65*r)
    arm = (np.abs(theta_base) < width/2)
    rings = np.zeros_like(r, dtype=float)
    for k in range(1, spikes+1):
        center = k/(spikes+1.0)
        rings += np.exp(- ((r - center)**2) * (sharpness*2.2))
    rings = (rings / rings.max()) if rings.max() > 0 else rings
    ring_mask = (rings > (1.0 - ring_strength))
    noise = (rng.random(r.shape) < density*(0.35 + 0.65*(1-r)))
    edge = (np.abs(theta_base) > (width*0.4)) & (np.abs(theta_base) < (width*0.5))
    pattern = arm | (ring_mask & arm) | (edge) | (noise & arm)
    return pattern

def snowflake_d6(size=64, params=None, seed=None):
    if params is None: params = {}
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=int)
    cx = (size-1)/2.0; cy = (size-1)/2.0
    y, x = np.mgrid[0:size, 0:size]
    dx = (x - cx) / (size/2.0); dy = (y - cy) / (size/2.0)
    r = np.sqrt(dx*dx + dy*dy); theta = np.arctan2(dy, dx)
    inside = r <= 0.96
    theta_base, _ = _reflect_to_base(theta)
    jitter = params.get("jitter", 0.03)
    r_j = np.clip(r + (rng.normal(0, jitter, r.shape)), 0, 1.2)
    th_j = theta_base + rng.normal(0, jitter*0.3, theta_base.shape)
    mask = base_wedge_pattern(r_j, th_j, params or {}, rng)
    grid[inside & mask] = 1
    for k in range(1, 6):
        ang = k * np.pi/3
        cos, sin = np.cos(ang), np.sin(ang)
        xrot = cos*dx - sin*dy; yrot = sin*dx + cos*dy
        r2 = np.sqrt(xrot*xrot + yrot*yrot)
        th2 = np.arctan2(yrot, xrot)
        th2b, _ = _reflect_to_base(th2)
        maskk = base_wedge_pattern(np.clip(r2,0,1.2), th2b, params or {}, np.random.default_rng(rng.integers(1e9)))
        grid[(r2 <= 0.96) & maskk] = 1
    return grid

# =========================
# Knit helpers
# =========================
def rows_from_pattern(pattern):
    translate = {0:"K",1:"P"}
    rows = []
    for i in range(pattern.shape[0]):
        seq = [translate[int(x)] for x in (pattern[i] if i%2==0 else pattern[i][::-1])]
        rows.append(f"Row {i+1:02d}: " + " ".join(seq))
    return rows

def export_pdf(pattern, title="Knit Chart"):
    bio = BytesIO()
    with PdfPages(bio) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.05,0.35,0.9,0.6])
        ax.imshow(1-pattern, cmap='gray', interpolation='nearest')
        ax.set_xticks(range(-1, pattern.shape[1]))
        ax.set_yticks(range(-1, pattern.shape[0]))
        ax.grid(True, linewidth=0.3)
        ax.set_title(title)
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                ax.text(j, i, "P" if pattern[i,j]==1 else "K", ha='center', va='center', fontsize=6)
        pdf.savefig(fig); plt.close(fig)
        # rows page(s)
        rows = rows_from_pattern(pattern)
        fig2 = plt.figure(figsize=(8.27, 11.69))
        ax2 = fig2.add_axes([0.05,0.05,0.9,0.9])
        ax2.axis('off')
        ax2.text(0.5, 0.97, "Row-by-Row Instructions", ha='center', fontsize=14)
        y = 0.92
        for r in rows:
            ax2.text(0.02, y, r, fontsize=8, family='monospace')
            y -= 0.03
            if y < 0.05:
                pdf.savefig(fig2); plt.close(fig2)
                fig2 = plt.figure(figsize=(8.27, 11.69))
                ax2 = fig2.add_axes([0.05,0.05,0.9,0.9])
                ax2.axis('off')
                ax2.text(0.5, 0.97, "Row-by-Row Instructions (cont.)", ha='center', fontsize=14)
                y = 0.92
        pdf.savefig(fig2); plt.close(fig2)
    bio.seek(0)
    return bio

# =========================
# Sweater mockup
# =========================
def render_sweater_mockup(pattern):
    """
    Render a simple sweater mockup with the pattern tiled on the torso area.
    We draw a rounded rectangle body + sleeves and fill torso with pattern.
    """
    H, W = pattern.shape
    fig = plt.figure(figsize=(5,6))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 100); ax.set_ylim(0, 120)

    # Body outline (rounded rectangle)
    body = FancyBboxPatch((15,20), 70, 80, boxstyle="round,pad=2,rounding_size=15", fill=False, linewidth=2)
    ax.add_patch(body)

    # Sleeves (simple rectangles)
    ax.add_patch(FancyBboxPatch((0,55), 20, 40, boxstyle="round,pad=1,rounding_size=8", fill=False, linewidth=2))
    ax.add_patch(FancyBboxPatch((80,55), 20, 40, boxstyle="round,pad=1,rounding_size=8", fill=False, linewidth=2))

    # Neck opening
    ax.add_patch(Circle((50, 100), radius=8, fill=False, linewidth=2))

    # Torso pattern area (tile pattern inside body bounds 17..83 x 22..98)
    # Convert pattern to an image array 0/1 -> colors via imshow grayscale
    tile_scale = 2  # tile count scale
    tiled = np.tile(pattern, (tile_scale, tile_scale))
    # Clip to a fixed display box
    ax.imshow(1-tiled, extent=(17, 83, 22, 98), interpolation='nearest', origin='lower', cmap='gray')
    ax.set_title("Sweater Mockup Preview", pad=8)
    return fig

# =========================
# UI
# =========================
st.title("🧶 Knitting Pattern Studio")

mode = st.selectbox("Pattern Type", ["Classic Motif (tiling symmetry)", "Nordic Motif (template)", "D6 True Snowflake (6-fold)"])

if mode == "Classic Motif (tiling symmetry)":
    col = st.columns(2)
    with col[0]:
        motif = st.selectbox("Motif", list(MOTIFS.keys())[:-1], index=0)
        symmetry = st.selectbox("Symmetry", ["quad","horizontal","vertical"], index=0)
        scale = st.slider("Scale", 1, 6, 2)
        mix = st.checkbox("Mix random variation", value=False)
        seed = st.number_input("Seed (0=random)", value=0, step=1)
    pattern = generate_classic(motif, symmetry, scale, mix, int(seed))
    # Show chart
    fig = plt.figure(figsize=(5,5)); ax = fig.add_axes([0,0,1,1])
    ax.imshow(1-pattern, cmap='gray', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{motif} ({pattern.shape[0]}×{pattern.shape[1]})")
    st.pyplot(fig)
    # Mockup
    #st.pyplot(render_sweater_mockup(pattern))
    pdf = export_pdf(pattern, f"{motif} Knit Chart ({pattern.shape[0]}×{pattern.shape[1]})")
    st.download_button("Download PDF", data=pdf.getvalue(), file_name="knit_chart.pdf", mime="application/pdf")

elif mode == "Nordic Motif (template)":
    col = st.columns(2)
    with col[0]:
        symmetry = st.selectbox("Symmetry", ["quad","horizontal","vertical"], index=0)
        scale = st.slider("Scale", 1, 6, 2)
    pattern = generate_classic("nordic_snow", symmetry, scale, mix_random=False, seed=0)
    fig = plt.figure(figsize=(5,5)); ax = fig.add_axes([0,0,1,1])
    ax.imshow(1-pattern, cmap='gray', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"Nordic Snowflake ({pattern.shape[0]}×{pattern.shape[1]})")
    st.pyplot(fig)
    #st.pyplot(render_sweater_mockup(pattern))
    pdf = export_pdf(pattern, f"Nordic Snowflake Knit Chart ({pattern.shape[0]}×{pattern.shape[1]})")
    st.download_button("Download PDF", data=pdf.getvalue(), file_name="nordic_snowflake_knit_chart.pdf", mime="application/pdf")

else:  # D6 True Snowflake
    size = st.slider("Grid size", 32, 128, 64, step=4)
    spikes = st.slider("Spikes (rings)", 1, 8, 5)
    sharp = st.slider("Sharpness", 6.0, 24.0, 14.0)
    density = st.slider("Granular density", 0.0, 1.0, 0.35)
    ring_strength = st.slider("Ring strength", 0.0, 1.0, 0.55)
    jitter = st.slider("Jitter", 0.0, 0.10, 0.03)
    seed = st.number_input("Seed (0=random)", value=7, step=1)

    params = dict(spikes=spikes, sharpness=sharp, density=density, ring_strength=ring_strength, jitter=jitter)
    pattern = snowflake_d6(size=size, params=params, seed=None if seed==0 else int(seed))

    fig = plt.figure(figsize=(5,5)); ax = fig.add_axes([0,0,1,1])
    ax.imshow(1-pattern, cmap='gray', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"D6 Snowflake ({size}×{size})")
    st.pyplot(fig)
    #st.pyplot(render_sweater_mockup(pattern))
    pdf = export_pdf(pattern, f"D6 Snowflake Knit Chart ({size}×{size})")
    st.download_button("Download PDF", data=pdf.getvalue(), file_name="d6_snowflake_knit_chart.pdf", mime="application/pdf")

st.markdown("**Legend:** 0=K (knit), 1=P (purl). Reading: odd rows →, even rows ←.")
