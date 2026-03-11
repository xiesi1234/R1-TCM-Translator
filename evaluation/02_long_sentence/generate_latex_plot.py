#!/usr/bin/env python3
"""
Generate LaTeX pgfplots figure code (BLEU by length bucket).
"""

import json
import math
import os
import numpy as np


def resolve_data_file(script_dir: str) -> str:
    candidates = [
        os.path.join(script_dir, "bleu_by_length_results.json"),
        os.path.join(os.path.dirname(script_dir), "bleu_by_length_results.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "bleu_by_length_results.json not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def bucket_label(bucket_id: int) -> str:
    if bucket_id < 10:
        low = bucket_id * 10 + 1
        high = (bucket_id + 1) * 10
        return f"{low}--{high}"
    return "101+"


def calculate_avg_bleu(bucket_bleu: dict, bucket_ids: list[int]) -> list[float]:
    avg_bleu = []
    for bucket in bucket_ids:
        values = []
        for sub in ["_1", "_2", "_3"]:
            key = f"{bucket}{sub}"
            if key in bucket_bleu:
                values.append(bucket_bleu[key])
        avg_bleu.append(round(float(np.mean(values)), 1) if values else 0.0)
    return avg_bleu


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = resolve_data_file(script_dir)
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    first_model = next(iter(data.keys()))
    bucket_ids = sorted(int(k) for k in data[first_model]["bucket_counts"].keys())
    x_centers = [5 + 10 * i for i in bucket_ids]
    x_labels = [bucket_label(i) for i in bucket_ids]
    counts = [int(data[first_model]["bucket_counts"].get(str(i), 0)) for i in bucket_ids]

    models = [
        ("Qwen3-0.6B-SFT", "0.6B (SFT)", "mySalmon", "triangle*"),
        ("Qwen3-8B-checkpoint-1000", "8B (SFT+GRPO)", "myBlue", "square*"),
        ("Qwen3-0.6B-checkpoint-4000", "0.6B (SFT+GRPO)", "myOrange", "*"),
    ]
    models = [m for m in models if m[0] in data]
    if not models:
        raise RuntimeError("Input JSON does not include expected model keys.")

    all_bleus = []
    for model_name, _, _, _ in models:
        all_bleus.extend(calculate_avg_bleu(data[model_name]["bucket_bleu"], bucket_ids))
    max_bleu = max(10, int(math.ceil(max(all_bleus) / 10.0) * 10))
    ymax_count = int(max(counts) * 1.2) if counts else 100

    xticks = ",".join(str(x) for x in x_centers)
    xticklabels = ",".join(x_labels)
    xmax = max(x_centers) + 5 if x_centers else 60

    latex_content = r"""\documentclass[tikz,border=1cm]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

% Color definitions
\definecolor{mySalmon}{HTML}{EA3323}
\definecolor{myBlue}{HTML}{241DED}
\definecolor{myOrange}{HTML}{FF8C00}
\definecolor{myGreen}{HTML}{458832}

\begin{document}

\begin{tikzpicture}[yshift=-1.5cm]
    % Bar chart (right axis) - drawn first (background layer)
    \begin{axis}[
        width=12cm, height=6cm,
        axis y line*=right,
        axis x line*=bottom,
        xlabel=\textbf{Source Length (characters)},
        ylabel=\textbf{\textcolor{myGreen}{Count of Samples}},
        ybar,
        bar width=10pt,
"""
    latex_content += f"        xmin=0, xmax={xmax},\n"
    latex_content += f"        xtick={{{xticks}}},\n"
    latex_content += f"        xticklabels={{{xticklabels}}},\n"
    latex_content += r"""        ymin=0, ymax=""" + f"{ymax_count},\n"
    latex_content += r"""        ytick style={draw=none},
    ]
        \addplot[
            fill=myGreen!80,
            draw=none,
        ] coordinates {
"""

    for x, c in zip(x_centers, counts):
        latex_content += f"            ({x},{c})\n"
    latex_content += r"""        };

"""
    for x, c in zip(x_centers, counts):
        y_pos = int(c + max(10, ymax_count * 0.02))
        latex_content += f"        \\node at (axis cs:{x},{y_pos}) {{{c}}};\n"

    latex_content += r"""
    \end{axis}

    % Line chart (left axis) - drawn second (foreground layer)
    \begin{axis}[
        width=12cm, height=6cm,
        axis y line*=left,
        axis x line*=none,
        ylabel=\textbf{BLEU},
"""
    latex_content += f"        xmin=0, xmax={xmax},\n"
    latex_content += f"        ymin=0, ymax={max_bleu},\n"
    latex_content += r"""        xtick=\empty,
        grid=major,
        grid style={dashed, gray!30},
        clip=false,
        legend style={
            at={(0.5,1.4)},
            anchor=south,
            legend columns=3,
            draw=black,
            thick,
            fill=white,
        }
    ]
"""

    for model_name, label, color, mark in models:
        bleus = calculate_avg_bleu(data[model_name]["bucket_bleu"], bucket_ids)
        coords = " ".join([f"({x},{b})" for x, b in zip(x_centers, bleus)])
        line_width = "thick" if model_name == "Qwen3-8B-checkpoint-1000" else "semithick"
        anchor = "below" if mark == "square*" else "above"

        latex_content += f"        % --- {label} ---\n"
        latex_content += r"""        \addplot+[
"""
        latex_content += f"            color={color},\n"
        latex_content += f"            mark={mark},\n"
        latex_content += r"""            mark options={fill=white, scale=1.5},
"""
        latex_content += f"            {line_width},\n"
        latex_content += r"""        ] coordinates {
"""
        latex_content += f"            {coords}\n"
        latex_content += r"""        };
"""
        latex_content += f"        \\addlegendentry{{{label}}}\n\n"

        for x, b in zip(x_centers, bleus):
            latex_content += (
                f"        \\node[{anchor}] at (axis cs:{x},{b}) "
                f"{{\\small\\bfseries\\textcolor{{{color}}}{{{b:.1f}}}}};\n"
            )
        latex_content += "\n"

    latex_content += r"""    \end{axis}
\end{tikzpicture}

\end{document}
"""

    output_file = os.path.join(script_dir, "grpo_comparison.tex")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"Loaded data file: {data_file}")
    print(f"LaTeX file generated: {output_file}")
    print("\nCompile command:")
    print("  pdflatex grpo_comparison.tex")


if __name__ == "__main__":
    main()
