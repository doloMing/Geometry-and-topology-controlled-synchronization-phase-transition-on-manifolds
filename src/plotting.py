import matplotlib.pyplot as plt
import seaborn as sns


def set_paper_style() -> None:
    """Use a restrained plotting style suitable for manuscript figures."""
    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )


def save_figure(fig: plt.Figure, path_without_suffix) -> None:
    """Save both PDF and PNG versions of a figure."""
    fig.savefig(f"{path_without_suffix}.pdf")
    fig.savefig(f"{path_without_suffix}.png")

