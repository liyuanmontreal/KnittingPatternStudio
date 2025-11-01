import streamlit as st
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def mirror_quad(seed):
    import numpy as np
    top = np.concatenate([seed, np.fliplr(seed)], axis=1)
    bottom = np.flipud(top)
    return np.concatenate([top, bottom], axis=0)

def upscale(grid, scale):
    import numpy as np
    return np.kron(grid, np.ones((scale, scale), dtype=int))

MOTIFS = {
    "diamond": np.array([[0,0,0,1,0,0,0,0],
                         [0,0,1,1,1,0,0,0],
                         [0,1,1,1,1,1,0,0],
                         [1,1,1,1,1,1,1,0],
                         [0,1,1,1,1,1,0,0],
                         [0,0,1,1,1,0,0,0],
                         [0,0,0,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0]], dtype=int),
    "snow":    np.array([[0,0,1,0,0,1,0,0],
                         [0,1,1,1,1,1,1,0],
                         [1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1],
                         [0,1,1,1,1,1,1,0],
                         [0,0,1,0,0,1,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0]], dtype=int),
    "wave":    np.array([[0,0,1,1,0,0,1,1],
                         [0,1,1,0,0,1,1,0],
                         [1,1,0,0,1,1,0,0],
                         [1,0,0,1,1,0,0,1],
                         [1,0,0,1,1,0,0,1],
                         [1,1,0,0,1,1,0,0],
                         [0,1,1,0,0,1,1,0],
                         [0,0,1,1,0,0,1,1]], dtype=int),
    "heart":   np.array([[0,1,1,0,0,1,1,0],
                         [1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1],
                         [0,1,1,1,1,1,1,0],
                         [0,0,1,1,1,1,0,0],
                         [0,0,0,1,1,0,0,0],
                         [0,0,0,0,0,0,0,0]], dtype=int),
}

def generate_pattern(motif="snow", mix_random=True, symmetry="quad", scale=2, seed=None):
    rng = np.random.default_rng(seed)
    base = MOTIFS[motif]
    if mix_random:
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

def to_knit_instructions(pattern):
    translate = {0: "K", 1: "P"}
    rows = []
    h, w = pattern.shape
    for i in range(h):
        r = pattern[i]
        if i % 2 == 0:
            seq = [translate[int(x)] for x in r]
            direction = "→"
        else:
            seq = [translate[int(x)] for x in r[::-1]]
            direction = "←"
        rows.append(f"Row {i+1:02d} {direction}: " + " ".join(seq))
    return rows

def draw_grid_ax(ax, pattern, show_symbols=True, title="Knit Chart"):
    ax.imshow(1 - pattern, cmap='gray', interpolation='nearest')
    ax.set_xticks(range(-1, pattern.shape[1]))
    ax.set_yticks(range(-1, pattern.shape[0]))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True, linewidth=0.3)
    ax.set_title(title)
    if show_symbols:
        h, w = pattern.shape
        for i in range(h):
            for j in range(w):
                ch = "P" if pattern[i, j] == 1 else "K"
                ax.text(j, i, ch, ha='center', va='center', fontsize=6)

def make_pdf(pattern, motif):
    bio = BytesIO()
    with PdfPages(bio) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.08, 0.35, 0.84, 0.6])
        draw_grid_ax(ax, pattern, True, f"Knit Chart — {motif}")
        axm = fig.add_axes([0.08, 0.1, 0.84, 0.2])
        axm.axis('off')
        axm.text(0.01, 0.8, "Notes", fontsize=12, weight='bold')
        axm.text(0.01, 0.55, f"Legend: 0 → K, 1 → P  |  Reading: odd rows →, even rows ←  |  Size: {pattern.shape[0]}×{pattern.shape[1]}",
                 fontsize=10, va='top')
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        rows = to_knit_instructions(pattern)
        fig2 = plt.figure(figsize=(8.27, 11.69))
        ax2 = fig2.add_axes([0.06, 0.06, 0.88, 0.88])
        ax2.axis('off')
        ax2.text(0.5, 0.97, "Row-by-Row Instructions", ha='center', va='top', fontsize=14, weight='bold')
        x_positions = [0.02, 0.52]
        y = 0.92; col = 0
        for line in rows:
            ax2.text(x_positions[col], y, line, fontsize=8, va='top', family='monospace')
            y -= 0.028
            if y < 0.05:
                col += 1; y = 0.92
                if col > 1:
                    pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)
                    fig2 = plt.figure(figsize=(8.27, 11.69))
                    ax2 = fig2.add_axes([0.06, 0.06, 0.88, 0.88])
                    ax2.axis('off')
                    ax2.text(0.5, 0.97, "Row-by-Row Instructions (cont.)", ha='center', va='top', fontsize=14, weight='bold')
                    x_positions = [0.02, 0.52]; y = 0.92; col = 0
        pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)
    bio.seek(0)
    return bio

st.title("🧶 Knitting Pattern Studio — PDF Export")
motif = st.selectbox("Motif", list(MOTIFS.keys()), index=1)
sym = st.selectbox("Symmetry", ["quad", "horizontal", "vertical"], index=0)
scale = st.slider("Scale (enlarge grid cells)", 1, 6, 2)
mix = st.checkbox("Mix random variation", value=True)
seed = st.number_input("Random seed (optional, 0=none)", value=0, step=1)

if st.button("Generate"):
    pattern = generate_pattern(motif, mix, sym, scale, int(seed) if seed!=0 else None)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    draw_grid_ax(ax, pattern, True, f"{motif} ({pattern.shape[0]}×{pattern.shape[1]})")
    st.pyplot(fig)
    pdf_bytes = make_pdf(pattern, motif)
    st.download_button("Download PDF", data=pdf_bytes.getvalue(), file_name="knit_chart.pdf", mime="application/pdf")

st.markdown("Legend: **0=K (knit)**, **1=P (purl)**. Reading: odd rows →, even rows ←.")