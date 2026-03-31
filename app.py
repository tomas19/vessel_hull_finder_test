import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "vessel_hulls_combined.csv"

DISPLAY_COLS = [
    "Vessel ID",
    "Vessel Type",
    "Vessel Type Detail",
    "Catalogue",
    "Hull File",
    "LOA [m]",
    "Beam [m]",
    "Height [m]",
    "Scale LOA",
    "Scale Beam",
    "Scale Height",
    "Uniform Scale",
    "Scale Spread (CV)",
    "Similarity Distance",
]

FORMAT_DICT = {
    "LOA [m]": "{:.1f}",
    "Beam [m]": "{:.1f}",
    "Height [m]": "{:.1f}",
    "Scale LOA": "{:.3f}",
    "Scale Beam": "{:.3f}",
    "Scale Height": "{:.3f}",
    "Uniform Scale": "{:.3f}",
    "Scale Spread (CV)": "{:.4f}",
    "Similarity Distance": "{:.4f}",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Breath [m]": "Beam [m]"})
    for col in ["LOA [m]", "Beam [m]", "Height [m]"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_similarity(
    df: pd.DataFrame,
    target_loa: float,
    target_beam: float,
    target_height: float,
) -> pd.DataFrame:
    """Return a copy of *df* with scaling factors and a similarity score."""
    out = df.copy()

    out["Scale LOA"] = target_loa / out["LOA [m]"]
    out["Scale Beam"] = target_beam / out["Beam [m]"]
    out["Scale Height"] = target_height / out["Height [m]"]

    out["Uniform Scale"] = np.cbrt(
        out["Scale LOA"] * out["Scale Beam"] * out["Scale Height"]
    )

    scales = out[["Scale LOA", "Scale Beam", "Scale Height"]]
    out["Scale Spread (CV)"] = scales.std(axis=1) / scales.mean(axis=1)

    out["Rel Diff LOA"] = (target_loa - out["LOA [m]"]) / target_loa
    out["Rel Diff Beam"] = (target_beam - out["Beam [m]"]) / target_beam
    out["Rel Diff Height"] = (target_height - out["Height [m]"]) / target_height
    out["Similarity Distance"] = np.sqrt(
        out["Rel Diff LOA"] ** 2
        + out["Rel Diff Beam"] ** 2
        + out["Rel Diff Height"] ** 2
    )

    return out


def style_table(
    df: pd.DataFrame, vmin: float, vmax: float
) -> "pd.io.formats.style.Styler":
    """Apply a green-to-red background colormap on Similarity Distance."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gyr", ["#2ecc71", "#f1c40f", "#e74c3c"]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _row_color(row):
        val = row["Similarity Distance"]
        rgba = cmap(norm(val))
        bg = f"background-color: rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.35)"
        return [bg] * len(row)

    cols = [c for c in DISPLAY_COLS if c in df.columns]
    fmt = {k: v for k, v in FORMAT_DICT.items() if k in cols}
    return df[cols].style.apply(_row_color, axis=1).format(fmt)


def build_tooltip_html(row: pd.Series) -> str:
    """Build an HTML tooltip string with all CSV fields for a vessel."""
    parts = []
    for col, val in row.items():
        parts.append(f"<b>{col}</b>: {val}")
    return "<br>".join(parts)


def render_detail_cards(df: pd.DataFrame, label: str) -> None:
    """Show summary statistics for a group of vessels."""
    st.markdown(f"**{label}** — {len(df)} vessel(s)")
    if df.empty:
        return
    num_cols = [
        "LOA [m]",
        "Beam [m]",
        "Height [m]",
        "Similarity Distance",
        "Scale LOA",
        "Scale Beam",
        "Scale Height",
        "Uniform Scale",
        "Scale Spread (CV)",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    stats = df[num_cols].describe().loc[["min", "mean", "max"]]
    st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)


def render_vessel_details(
    results: pd.DataFrame, raw_df: pd.DataFrame, key_prefix: str
) -> None:
    """Expandable per-vessel detail — collapsible via expander."""
    vessel_ids = results["Vessel ID"].tolist()
    with st.expander("Inspect full parameters for a vessel", expanded=False):
        selected = st.selectbox(
            "Select vessel:",
            options=["— select —"] + vessel_ids,
            key=f"detail_{key_prefix}",
        )
        if selected != "— select —":
            row = raw_df[raw_df["Vessel ID"] == selected].iloc[0]
            detail = (
                row.to_frame("Value")
                .reset_index()
                .rename(columns={"index": "Parameter"})
            )
            st.dataframe(detail, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Vessel Hull Finder", layout="wide")
    st.title("🚢 Vessel Hull Finder")
    st.markdown(
        "Find the closest hull model from the catalogue and compute the "
        "scaling factors needed to match your target vessel dimensions."
    )

    df_raw = load_data()

    # --- Sidebar inputs ---
    st.sidebar.header("Target vessel")
    target_loa = st.sidebar.number_input(
        "LOA [m]", min_value=1.0, value=200.0, step=1.0
    )
    target_beam = st.sidebar.number_input(
        "Beam [m]", min_value=0.1, value=32.0, step=0.5
    )
    target_height = st.sidebar.number_input(
        "Height [m]", min_value=0.1, value=18.0, step=0.5
    )

    vessel_types = sorted(df_raw["Vessel Type"].dropna().unique(), key=str.lower)
    selected_type = st.sidebar.selectbox("Vessel Type", options=vessel_types)

    # --- Split into same type / other types ---
    df_complete = df_raw.dropna(subset=["LOA [m]", "Beam [m]", "Height [m]"])
    if df_complete.empty:
        st.warning("No vessels with complete dimensions in the database.")
        return

    all_results = compute_similarity(
        df_complete, target_loa, target_beam, target_height
    )

    mask_same = all_results["Vessel Type"].str.lower() == selected_type.lower()
    df_same = (
        all_results[mask_same].sort_values("Similarity Distance").reset_index(drop=True)
    )
    df_other = (
        all_results[~mask_same]
        .sort_values("Similarity Distance")
        .reset_index(drop=True)
    )
    df_same.index += 1
    df_other.index += 1

    # Common color norm across both tables
    vmin = all_results["Similarity Distance"].min()
    vmax = all_results["Similarity Distance"].max()

    # --- Table 1: Same vessel type ---
    st.subheader(f"Same class — {selected_type}")
    if df_same.empty:
        st.info(f"No vessels of type '{selected_type}' with complete dimensions.")
    else:
        st.dataframe(
            style_table(df_same, vmin, vmax), use_container_width=True, height=400
        )
        render_vessel_details(df_same, df_raw, key_prefix="same")

    # --- Table 2: Other vessel types ---
    st.subheader("Other classes")
    if df_other.empty:
        st.info("No other vessel types with complete dimensions.")
    else:
        st.dataframe(
            style_table(df_other, vmin, vmax), use_container_width=True, height=500
        )
        render_vessel_details(df_other, df_raw, key_prefix="other")

    # --- Summary statistics ---
    st.divider()
    st.subheader("Summary statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        render_detail_cards(df_same, f"Same class ({selected_type})")
    with col_b:
        render_detail_cards(df_other, "Other classes")

    # --- Best match card ---
    best = all_results.sort_values("Similarity Distance").iloc[0]
    st.divider()
    st.subheader("Best overall match")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vessel ID", best["Vessel ID"])
    col2.metric("Type", best["Vessel Type Detail"])
    col3.metric("Catalogue", best["Catalogue"])

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Scale LOA", f"{best['Scale LOA']:.3f}")
    col5.metric("Scale Beam", f"{best['Scale Beam']:.3f}")
    col6.metric("Scale Height", f"{best['Scale Height']:.3f}")
    col7.metric("Uniform Scale", f"{best['Uniform Scale']:.3f}")

    st.caption(
        "**Uniform Scale** = geometric mean of the three individual scale factors. "
        "**Scale Spread (CV)** = coefficient of variation of the three scales — "
        "lower means the hull proportions match better (0 = perfect geometric similarity). "
        "Row colours: 🟢 green = close match → 🔴 red = poor match."
    )


if __name__ == "__main__":
    main()
