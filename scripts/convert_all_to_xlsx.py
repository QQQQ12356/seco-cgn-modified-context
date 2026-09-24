"""
批量将 eval_result/ 下的所有 mrqa_inference_metrics.json 提取为 xlsx。

与参考脚本 convert.py 的区别：
- 不再单个文件转单个 xlsx，而是扫描整个目录，把多个实验的"关键"指标汇总到
  一个 xlsx 中，并用若干"来源列"明确区分每一行来自哪个文件 / 哪个模型 / 哪种压缩配置。

输出 sheets:
  summary        : 每行一个实验，含来源标识 + summary 指标（EM/F1/id/ood）
  subset_summary : 每行一个实验 x 一个子集，长表，含来源标识 + 子集指标

来源标识列（区分"这些数据来自哪个文件/模型/配置"）:
  source_file        : mrqa_inference_metrics.json 的相对路径
  model_name_or_path : eval_manifest.json 的 model_name_or_path（缺失时取路径中的模型目录名）
  compressor_version : 压缩器版本（路径第一层，如 context_adaptive_v1）
  model_dir          : 路径中的模型目录名（如 Llama-3.2-1B-Instruct）
  config_dir         : 路径中的配置目录名（如 ratio_16/.../cr16_bs1_samples2000_cp20000）
  compress_ratio     : manifest.route.compress_ratio 或从 config_dir 推断 (cr<数>)
  samples / eval_samples / seed / checkpoint : manifest 字段

用法:
  python scripts/convert_all_to_xlsx.py [eval_result_dir] [-o output.xlsx]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

METRICS_NAME = "mrqa_inference_metrics.json"
MANIFEST_NAME = "eval_manifest.json"

# 与参考脚本一致的子数据集展示顺序
SUBSET_ORDER = [
    "DROP",
    "BioASQ",
    "DuoRC.ParaphraseRC",
    "TextbookQA",
    "RelationExtraction",
    "RACE",
    "SQuAD",
    "NewsQA",
    "TriviaQA-web",
    "SearchQA",
    "HotpotQA",
    "NaturalQuestionsShort",
]


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_make(v, default):
    """构造 list 但只接受非空值。"""
    if v in (None, "", []):
        return default
    return v


def _infer_ratio_from_config(config_dir: str):
    """从配置目录名中提取 compress_ratio，如 cr16_bs1_... -> 16。"""
    m = re.search(r"cr(\d+)", config_dir)
    return int(m.group(1)) if m else None


def _build_source_columns(metrics_path: Path, metrics: dict, manifest: dict, root: Path) -> dict:
    """构造用于区分"这些数据来自哪个文件/模型"的来源列。

    metrics_path 与 root 均为绝对路径。来源列基于"相对 root 的路径"解析，
    结构约定为: <root>/<compressor_version>/[<model_dir>/]<config...>/mrqa_inference_metrics.json
    """
    rel = metrics_path.relative_to(root).as_posix()
    rel_parts = metrics_path.relative_to(root).parts

    # 版本层 = 相对路径第一层；其余为 模型目录 + 配置目录 + 文件名
    compressor_version = rel_parts[0] if len(rel_parts) > 0 else ""
    # 配置目录：去掉 version 层与最后的文件名层
    config_dir = ""
    if len(rel_parts) >= 3:
        config_dir = "/".join(rel_parts[1:-1])

    route = manifest.get("route", {}) or {}
    model = manifest.get("model_name_or_path", "")

    ratio = route.get("compress_ratio")
    if ratio is None:
        ratio = _infer_ratio_from_config(config_dir)

    cols = {
        "source_file": rel,
        "model_name_or_path": _safe_make(model, None),
        "compressor_version": _safe_make(compressor_version, None),
        "config_dir": _safe_make(config_dir, None),
        "compress_ratio": ratio,
        "samples": metrics["summary"].get("total_samples"),
        "eval_samples": manifest.get("eval_samples"),
        "num_rows_loaded": manifest.get("num_rows_loaded"),
        "seed": manifest.get("seed"),
        "checkpoint": manifest.get("checkpoint"),
    }
    return cols


def build_summary_rows(metrics_files, root):
    """每个实验一行 summary，来源列在前。"""
    rows = []
    for path in metrics_files:
        metrics = _read_json(path)
        manifest = {}
        mpath = path.with_name(MANIFEST_NAME)
        if mpath.exists():
            manifest = _read_json(mpath)

        row = _build_source_columns(path, metrics, manifest, root)
        # summary 指标
        for k, v in metrics["summary"].items():
            row[k] = v
        # route 里的关键超参，帮助区分不同压缩配置
        for key in ("selection_mode", "budget_mode", "context_budget_mode",
                    "query_slots", "relevance_dim", "raw_memory_fraction",
                    "context_boundary_mode", "context_novelty_weight"):
            if (manifest.get("route") or {}).get(key) is not None:
                row[key] = manifest["route"][key]
        rows.append(row)
    return rows


def build_subset_rows(metrics_files, root):
    """长表：每个实验 x 每个子集一行，来源列在前。"""
    rows = []
    for path in metrics_files:
        metrics = _read_json(path)
        manifest = {}
        mpath = path.with_name(MANIFEST_NAME)
        if mpath.exists():
            manifest = _read_json(mpath)

        base = _build_source_columns(path, metrics, manifest, root)
        subset_dict = metrics["subset_summary"]

        ordered = [s for s in SUBSET_ORDER if s in subset_dict]
        extras = sorted(set(subset_dict) - set(SUBSET_ORDER))
        for subset in ordered + extras:
            row = dict(base)
            row["subset"] = subset
            row.update(subset_dict[subset])  # count, avg_em, avg_f1
            rows.append(row)
    return rows


def write_xlsx(summary_rows, subset_rows, output_path: Path):
    summary_df = pd.DataFrame(summary_rows)
    subset_df = pd.DataFrame(subset_rows)

    # 把来源列排到最前
    src_cols = ["source_file", "model_name_or_path", "compressor_version",
                "config_dir", "compress_ratio", "samples", "eval_samples",
                "seed", "checkpoint"]

    def reorder(df):
        cols = src_cols + [c for c in df.columns if c not in src_cols]
        return df[[c for c in cols if c in df.columns]]

    summary_df = reorder(summary_df)
    subset_df = reorder(subset_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        subset_df.to_excel(writer, sheet_name="subset_summary", index=False)

    print(f"成功生成: {output_path}")
    print(f"  summary 行数: {len(summary_rows)}, subset_summary 行数: {len(subset_rows)}")


def main():
    parser = argparse.ArgumentParser(description="批量提取 eval_result 中所有 metrics json 为 xlsx")
    parser.add_argument("root", nargs="?", default="eval_result",
                        help="结果根目录（默认 eval_result）")
    parser.add_argument("-o", "--output", default="./xlsx/all_metrics.xlsx",
                        help="输出 xlsx 路径（默认 ./xlsx/all_metrics.xlsx）")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    metrics_files = sorted(root.rglob(METRICS_NAME))
    if not metrics_files:
        print(f"未找到任何 {METRICS_NAME} 文件于: {root}")
        return

    summary_rows = build_summary_rows(metrics_files, root)
    subset_rows = build_subset_rows(metrics_files, root)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(summary_rows, subset_rows, out)


if __name__ == "__main__":
    main()