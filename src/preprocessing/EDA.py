from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm


class SkinDetectorEDA:
    """Handle professional-grade Exploratory Data Analysis for image datasets.

    Analyze the distribution, metadata, and geometry of canine skin disease images.
    """

    def __init__(self, data_path: Path, report_path: Path) -> None:
        """Initialize the EDA tool with necessary paths.

        Args:
            data_path: Path object pointing to the 'raw' images directory.
            report_path: Path object where all outputs will be stored.

        """
        self.data_path: Final[Path] = data_path
        self.report_path: Final[Path] = report_path
        self.valid_exts: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp")
        self.report_path.mkdir(parents=True, exist_ok=True)

    def run_full_analysis(self) -> None:
        """Execute distribution analysis, geometry analysis, and CSV export."""
        print(f"[*] Starting EDA at: {self.data_path}")

        df_dist = self._analyze_distribution()
        df_dims = self._analyze_dimensions()

        if df_dist.empty or df_dims.empty:
            print("[-] Error: Dataset is empty or path is incorrect.")
            return

        # Export metadata to CSV
        csv_path = self.report_path / "dimensions_report.csv"
        df_dims.to_csv(csv_path, index=False)
        print(f"[+] Metadata CSV exported to: {csv_path}")

        self._plot_distribution(df_dist)
        self._plot_dimensions(df_dims)
        self._print_executive_summary(df_dist, df_dims)

    def _analyze_distribution(self) -> pd.DataFrame:
        """Count files per category and calculate percentages."""
        stats = []
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                files = [
                    f
                    for f in class_dir.iterdir()
                    if f.suffix.lower() in self.valid_exts
                ]
                if files:
                    stats.append({"Class": class_dir.name, "Count": len(files)})

        df = pd.DataFrame(stats).sort_values(by="Count", ascending=False)
        if not df.empty:
            df["Percentage (%)"] = (df["Count"] / df["Count"].sum() * 100).round(2)
        return df

    def _analyze_dimensions(self) -> pd.DataFrame:
        """Extract metadata (Filename, Class, W, H, AR) from images."""
        data = []
        image_files = [
            f for f in self.data_path.rglob("*") if f.suffix.lower() in self.valid_exts
        ]

        for img_path in tqdm(image_files, desc="Processing Image Metadata"):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    data.append(
                        {
                            "Filename": img_path.name,
                            "Class": img_path.parent.name,
                            "Width": w,
                            "Height": h,
                            "Aspect_Ratio": w / h,
                        }
                    )
            except (OSError, ValueError):
                continue
        return pd.DataFrame(data)

    def _plot_distribution(self, df: pd.DataFrame) -> None:
        """Generate a detailed bar chart with distribution labels."""
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            x="Count",
            y="Class",
            data=df,
            palette="viridis",
            hue="Class",
            legend=False,
        )

        total = df["Count"].sum()
        for i, (count, pct) in enumerate(
            zip(df["Count"], df["Percentage (%)"], strict=True)
        ):
            ax.text(
                count + (total * 0.005),
                i,
                f"{int(count)} ({pct}%)",
                va="center",
                fontweight="bold",
            )

        plt.title("Class Distribution", fontsize=16, pad=20)
        output = self.report_path / "distribution_detailed.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_dimensions(self, df: pd.DataFrame) -> None:
        """Create a jointplot with reference aspect ratio lines."""
        sns.set_theme(style="white")
        grid = sns.jointplot(
            data=df,
            x="Width",
            y="Height",
            hue="Class",
            kind="scatter",
            alpha=0.5,
            palette="magma",
            height=10,
        )

        max_dim = max(df["Width"].max(), df["Height"].max())
        x_ref = np.linspace(0, max_dim, 100)
        grid.ax_joint.plot(x_ref, x_ref, "r--", alpha=0.6, label="1:1 (Square)")
        grid.ax_joint.plot(
            x_ref, x_ref * 0.75, "g--", alpha=0.6, label="4:3 (Landscape)"
        )
        grid.ax_joint.plot(
            x_ref, x_ref * 1.33, "b--", alpha=0.6, label="3:4 (Portrait)"
        )

        grid.ax_joint.legend(loc="upper left")
        grid.figure.suptitle("Image Geometry Analysis", y=1.02, fontsize=16)

        output = self.report_path / "dimensions_detailed.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

    def _print_executive_summary(
        self, df_dist: pd.DataFrame, df_dims: pd.DataFrame
    ) -> None:
        """Print final scientific recommendations for preprocessing."""
        ir = df_dist["Count"].max() / df_dist["Count"].min()
        std_ar = df_dims["Aspect_Ratio"].std()

        print("\n" + "=" * 60)
        print("                EXECUTIVE EDA SUMMARY REPORT")
        print("=" * 60)
        print(f"Total Samples:      {df_dist['Count'].sum()}")
        print(f"Imbalance Ratio:    {ir:.2f}")
        print(f"AR Consistency:     {std_ar:.3f} (Std Dev)")
        print("-" * 60)
        print(f"[✓] Reports and CSV saved in: {self.report_path.resolve()}")
        print("=" * 60)


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent
    project_root = src_dir.parent

    raw_data_path = project_root / "data" / "raw"
    reports_path = src_dir / "reports"

    if not raw_data_path.exists():
        print(f"[-] Error: Could not find data at {raw_data_path}")
    else:
        eda = SkinDetectorEDA(raw_data_path, reports_path)
        eda.run_full_analysis()
