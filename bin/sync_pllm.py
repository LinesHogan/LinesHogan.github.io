#!/usr/bin/env python3
"""Synchronize bilingual blog posts, human-facing bilingual docs, and SVG illustrations from pLLM into LinesHogan.github.io.

Usage:
  python3 bin/sync_pllm.py --pllm-dir /path/to/pLLM
  python3 bin/sync_pllm.py   # shallow-clones https://github.com/LinesHogan/pLLM.git automatically
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

REPO_URL_DEFAULT = "https://github.com/LinesHogan/pLLM.git"
REPO_WEB_BASE = "https://github.com/LinesHogan/pLLM"
CST = datetime.timezone(datetime.timedelta(hours=8))

# Human-only doc categories (doc/agent/ is intentionally excluded)
DOC_CATEGORIES = {
    "tutorials": ("交互式入门教程 (00–05)", "Step-by-Step Tutorials (00–05)", "入门教程", "Tutorials"),
    "guides": ("用户指南", "User Guides", "用户指南", "Guides"),
    "howto": ("操作与上线指南", "How-To & Serving Guides", "操作指南", "How-To"),
    "explain": ("架构与机制解析", "Architecture & Design Explanations", "架构解析", "Architecture"),
    "reference": ("API 与配置参考手册", "API, Config & Support Reference", "参考手册", "Reference"),
    "examples": ("算法族示例与实测 (06–13)", "Algorithm Family Examples (06–13)", "算法示例", "Examples"),
}


def run_git(cwd: pathlib.Path, args: List[str]) -> str:
    res = subprocess.run(
        ["git", "-C", str(cwd)] + args,
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() if res.returncode == 0 else ""


def strip_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    if not text.startswith("---\n"):
        return meta, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return meta, text
    raw_fm = text[4:end]
    for line in raw_fm.splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[end + 5 :].lstrip("\n")


def github_slug(heading_text: str) -> str:
    clean = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", heading_text)
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    clean = clean.replace("`", "").replace("**", "").replace("*", "")
    clean = clean.strip().lower()
    out_chars: List[str] = []
    for ch in clean:
        if ch == " " or ch == "-":
            out_chars.append("-")
        elif ch == "_" or ch.isalnum() or ord(ch) > 127:
            if ch in "，。！？：；、“”‘’（）《》【】·—…—–/\\()[]{}:;,.!?\"'+~=<>&|$%@^#*":
                continue
            out_chars.append(ch)
    return "".join(out_chars).strip("-") or "section"


class SlugTracker:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.counts: Dict[str, int] = {}
        self.sec_counter = 0

    def register(self, raw_heading: str) -> Tuple[str, str]:
        base = github_slug(raw_heading)
        idx = self.counts.get(base, 0)
        self.counts[base] = idx + 1
        gh = base if idx == 0 else "{}-{}".format(base, idx)
        if re.match(r"^[a-z][a-z0-9_-]*$", gh):
            ascii_id = "{}{}".format(self.prefix, gh)
        else:
            self.sec_counter += 1
            ascii_id = "{}sec-{}".format(self.prefix, self.sec_counter)
        return gh, ascii_id


def yaml_quote(val: str) -> str:
    escaped = val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()
    return '"{}"'.format(escaped)


def html_attr_escape(val: str) -> str:
    return (
        val.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def normalize_doc_subpath(rel_posix: str) -> Optional[str]:
    """If `rel_posix` is inside `doc/` or `doc_en/` and is human-facing (not `agent/` or `dev/`), return subpath."""
    if rel_posix.startswith("doc/"):
        sub = rel_posix[len("doc/") :]
    elif rel_posix.startswith("doc_en/"):
        sub = rel_posix[len("doc_en/") :]
    else:
        return None
    if sub.startswith(("dev/", "agent/", "assets/")):
        return None
    p = pathlib.PurePosixPath(sub)
    if p.suffix not in (".md", ".ipynb"):
        return None
    return sub


def doc_url_for_rel(rel_posix: str) -> Optional[str]:
    sub = normalize_doc_subpath(rel_posix)
    if sub is None:
        return None
    if sub == "README.md":
        return "/pllm/docs/"
    p = pathlib.PurePosixPath(sub)
    stem = p.stem.lower()
    parent = str(p.parent)
    if parent == ".":
        return "/pllm/docs/{}/".format(stem)
    return "/pllm/docs/{}/{}/".format(parent.lower(), stem)


def doc_output_rel(sub: str) -> str:
    if sub == "README.md":
        return "index.md"
    p = pathlib.PurePosixPath(sub)
    stem = p.stem.lower()
    parent = str(p.parent)
    if parent == ".":
        return "{}.md".format(stem)
    return "{}/{}.md".format(parent.lower(), stem)


def notebook_to_raw_markdown(nb_path: pathlib.Path) -> str:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    parts: List[str] = []
    for cell in data.get("cells", []):
        ctype = cell.get("cell_type")
        raw_src = cell.get("source", "")
        src = "".join(raw_src) if isinstance(raw_src, list) else str(raw_src)
        src = src.rstrip()
        if not src:
            continue
        if ctype == "markdown":
            parts.append(src)
        elif ctype == "code":
            block = "```python\n{}\n```".format(src)
            outputs = cell.get("outputs", [])
            stdout_chunks: List[str] = []
            for out in outputs:
                if out.get("name") == "stdout" or out.get("output_type") == "stream":
                    txt = out.get("text", "")
                    stdout_chunks.append("".join(txt) if isinstance(txt, list) else str(txt))
            stdout_str = "".join(stdout_chunks).strip()
            if stdout_str:
                lines = stdout_str.splitlines()
                if len(lines) > 40:
                    stdout_str = "\n".join(lines[:38] + ["... (output truncated)"])
                block += "\n\n```text\n{}\n```".format(stdout_str)
            parts.append(block)
    return "\n\n".join(parts) + "\n"


def copy_static_assets(pllm_dir: pathlib.Path, site_dir: pathlib.Path) -> None:
    """Mirror pLLM/blog/assets and pLLM/doc/assets into assets/img/pllm/."""
    dest_root = site_dir / "assets" / "img" / "pllm"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    for rel_sub in ("blog/assets", "doc/assets"):
        src_dir = pllm_dir / rel_sub
        if src_dir.exists():
            target_dir = dest_root / rel_sub
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, target_dir)


def scan_doc_anchors(pllm_dir: pathlib.Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Pre-scan all human docs in both `doc/` (zh-) and `doc_en/` (en-) for heading anchors and H1 titles."""
    anchor_map: Dict[str, Dict[str, str]] = {}
    titles: Dict[str, str] = {}

    for root_name, id_prefix in (("doc", "zh-"), ("doc_en", "en-")):
        droot = pllm_dir / root_name
        if not droot.exists():
            continue
        for path in sorted(droot.rglob("*")):
            if not path.is_file() or ".ipynb_checkpoints" in path.parts:
                continue
            rel_posix = path.relative_to(pllm_dir).as_posix()
            if normalize_doc_subpath(rel_posix) is None:
                continue
            raw = (
                notebook_to_raw_markdown(path)
                if path.suffix == ".ipynb"
                else strip_frontmatter(path.read_text(encoding="utf-8"))[1]
            )
            tracker = SlugTracker(prefix=id_prefix)
            file_map: Dict[str, str] = {}
            fenced = False
            first_h1: Optional[str] = None
            for line in raw.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    fenced = not fenced
                    continue
                if fenced:
                    continue
                m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
                if m:
                    level = len(m.group(1))
                    htitle = m.group(2).strip()
                    if level == 1 and first_h1 is None:
                        first_h1 = re.sub(r"`([^`]+)`", r"\1", htitle)
                    gh, ascii_id = tracker.register(htitle)
                    file_map[gh] = ascii_id
            anchor_map[rel_posix] = file_map
            titles[rel_posix] = first_h1 or path.stem
    return anchor_map, titles


def resolve_relative_url(
    target: str,
    source_rel_posix: str,
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
    blog_url_by_stem: Dict[str, str],
    is_image: bool = False,
) -> str:
    if target.startswith(("http://", "https://", "mailto:", "/")) or target.startswith("#"):
        if target.startswith("#") and source_rel_posix in anchor_map:
            frag = target[1:]
            mapped = anchor_map[source_rel_posix].get(frag.lower(), anchor_map[source_rel_posix].get(frag))
            if mapped:
                return "#{}".format(mapped)
        return target

    if "#" in target:
        path_part, frag_part = target.split("#", 1)
    else:
        path_part, frag_part = target, ""

    src_parent = (pllm_dir / source_rel_posix).parent
    resolved = (src_parent / path_part).resolve()
    try:
        rel_target = resolved.relative_to(pllm_dir.resolve()).as_posix()
    except ValueError:
        return target

    # If a `doc_en/` file used a relative path that resolved into `doc_en/assets/...`, fall back to `doc/assets/...`
    if is_image and not resolved.exists() and rel_target.startswith("doc_en/"):
        alt_rel = "doc/" + rel_target[len("doc_en/") :]
        alt_resolved = (pllm_dir / alt_rel).resolve()
        if alt_resolved.exists():
            resolved = alt_resolved
            rel_target = alt_rel

    if is_image and resolved.exists() and resolved.is_file():
        dest = site_dir / "assets" / "img" / "pllm" / rel_target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, dest)
        return "/assets/img/pllm/{}".format(rel_target)

    if rel_target in ("README.md", "README.zh.md", "README.en.md"):
        return "/pllm/"
    if rel_target in ("blog/README.md", "blog/personal/README.md"):
        return "/pllm/#blog"
    if rel_target.startswith(("blog/zh/", "blog/en/")) and rel_target.endswith(".md"):
        stem = pathlib.PurePosixPath(rel_target).stem
        if stem in blog_url_by_stem:
            return blog_url_by_stem[stem]
    if rel_target.startswith("blog/personal/") and rel_target.endswith(".md"):
        p_stem = "personal-" + pathlib.PurePosixPath(rel_target).stem
        if p_stem in blog_url_by_stem:
            return blog_url_by_stem[p_stem]

    doc_url = doc_url_for_rel(rel_target)
    if doc_url is not None:
        if frag_part:
            tmap = anchor_map.get(rel_target, {})
            mapped_frag = tmap.get(frag_part.lower(), tmap.get(frag_part))
            if not mapped_frag:
                # Try sibling tree (`doc/` <-> `doc_en/`) if anchor came from translated link
                sibling = (
                    "doc/" + rel_target[len("doc_en/") :]
                    if rel_target.startswith("doc_en/")
                    else "doc_en/" + rel_target[len("doc/") :]
                )
                smap = anchor_map.get(sibling, {})
                mapped_frag = smap.get(frag_part.lower(), smap.get(frag_part))
            if mapped_frag:
                return "{}#{}".format(doc_url, mapped_frag)
        return doc_url

    # Non-human / repo files (e.g. `doc/agent/*`, `examples/*.py`, `measurements.toml`) point cleanly to GitHub
    if rel_target.startswith("doc_en/"):
        gh_check = pllm_dir / rel_target
        if not gh_check.exists():
            rel_target = "doc/" + rel_target[len("doc_en/") :]
    gh_url = "{}/blob/main/{}".format(REPO_WEB_BASE, rel_target)
    if frag_part:
        gh_url = "{}#{}".format(gh_url, frag_part)
    return gh_url


def convert_prose_chunk(
    chunk: str,
    source_rel_posix: str,
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
    blog_url_by_stem: Dict[str, str],
) -> str:
    chunk = chunk.replace("&nearr;", "↗")
    chunk = re.sub(r"\s*<!--\s*check-docs:[^>]*-->", "", chunk)

    code_tokens: List[str] = []

    def save_code(m: re.Match) -> str:
        span = m.group(0)
        if "{{" in span or "{%" in span:
            span = "{% raw %}" + span + "{% endraw %}"
        idx = len(code_tokens)
        code_tokens.append(span)
        return "\x00CODE{}\x00".format(idx)

    chunk = re.sub(r"(`+)([^`\n]+?)\1", save_code, chunk)

    def replace_img(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2).strip()
        new_url = resolve_relative_url(
            url, source_rel_posix, pllm_dir, site_dir, anchor_map, blog_url_by_stem, is_image=True
        )
        return '![{}]({}){{: style="display: block; max-width: 100%; height: auto; margin: 1.5rem auto;" }}'.format(
            alt, new_url
        )

    def replace_link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2).strip()
        new_url = resolve_relative_url(
            url, source_rel_posix, pllm_dir, site_dir, anchor_map, blog_url_by_stem, is_image=False
        )
        return "[{}]({})".format(label, new_url)

    chunk = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", replace_img, chunk)
    chunk = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", replace_link, chunk)

    math_tokens: List[str] = []

    def save_double_math(m: re.Match) -> str:
        expr = m.group(1)
        idx = len(math_tokens)
        math_tokens.append(expr)
        return "\x00MATH{}\x00".format(idx)

    chunk = re.sub(r"\$\$(.+?)\$\$", save_double_math, chunk, flags=re.S)

    def save_single_math(m: re.Match) -> str:
        expr = m.group(1)
        idx = len(math_tokens)
        math_tokens.append(expr)
        return "\x00MATH{}\x00".format(idx)

    chunk = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", save_single_math, chunk)

    if "{{" in chunk or "{%" in chunk:
        chunk = chunk.replace("{{", "{% raw %}{{{% endraw %}").replace("{%", "{% raw %}{%{% endraw %}")

    out_lines: List[str] = []
    for line in chunk.splitlines():
        is_table_row = line.lstrip().startswith("|")

        def restore_line_math(m: re.Match) -> str:
            idx = int(m.group(1))
            expr = math_tokens[idx]
            if is_table_row:
                expr = re.sub(r"(?<!\\)\|(?![|])", r"\\vert ", expr)
            if "{{" in expr or "{%" in expr:
                return "{% raw %}$$" + expr + "$${% endraw %}"
            return "$${}$$".format(expr)

        line = re.sub(r"\x00MATH(\d+)\x00", restore_line_math, line)
        line = re.sub(r"\x00CODE(\d+)\x00", lambda m: code_tokens[int(m.group(1))], line)
        out_lines.append(line)

    return "\n".join(out_lines)


def transform_markdown_document(
    raw_text: str,
    source_rel_posix: str,
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
    blog_url_by_stem: Dict[str, str],
    attach_kramdown_ids: bool = False,
    id_prefix: str = "",
    strip_leading_h1: bool = False,
) -> str:
    lines = raw_text.splitlines()
    tracker = SlugTracker(prefix=id_prefix)
    result_chunks: List[str] = []
    prose_buf: List[str] = []
    in_fence = False
    fence_marker = ""
    fence_indent = ""
    fence_buf: List[str] = []
    removed_h1 = False

    def flush_prose() -> None:
        if not prose_buf:
            return
        leading_nl = "\n" if prose_buf[0] == "" else ""
        trailing_nl = "\n" if prose_buf[-1] == "" else ""
        text_block = "\n".join(prose_buf).strip("\n")
        prose_buf.clear()
        if not text_block:
            result_chunks.append("")
            return
        converted = convert_prose_chunk(
            text_block,
            source_rel_posix,
            pllm_dir,
            site_dir,
            anchor_map,
            blog_url_by_stem,
        )
        result_chunks.append("{}{}{}".format(leading_nl, converted, trailing_nl))

    for line in lines:
        if not in_fence:
            fm = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", line)
            if fm:
                flush_prose()
                in_fence = True
                fence_indent = fm.group(1)
                fence_marker = fm.group(2)
                info = fm.group(3).strip()
                if info == "python gpu":
                    info = "python"
                elif info.startswith("text "):
                    info = "text"
                fence_buf = ["{}{}{}".format(fence_indent, fence_marker, info)]
                continue

            hm = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if hm:
                hashes, htitle = hm.group(1), hm.group(2).strip()
                _, ascii_id = tracker.register(htitle)
                if len(hashes) == 1 and strip_leading_h1 and not removed_h1:
                    removed_h1 = True
                    if attach_kramdown_ids:
                        prose_buf.append('<div id="{}"></div>'.format(ascii_id))
                    continue
                if attach_kramdown_ids:
                    prose_buf.append("{} {} {{#{}}}".format(hashes, htitle, ascii_id))
                else:
                    prose_buf.append(line)
                continue

            prose_buf.append(line)
        else:
            close_m = re.match(r"^(\s*)(`{3,}|~{3,})\s*$", line)
            if close_m and close_m.group(2)[0] == fence_marker[0] and len(close_m.group(2)) >= len(fence_marker):
                fence_buf.append(line)
                code_block = "\n".join(fence_buf)
                code_block = re.sub(r"^\s*#\s*check-docs:\s*skip:[^\n]*\n", "", code_block, flags=re.M)
                if "{{" in code_block or "{%" in code_block:
                    code_block = "{% raw %}\n" + code_block + "\n{% endraw %}"
                result_chunks.append(code_block)
                fence_buf = []
                in_fence = False
            else:
                fence_buf.append(line)

    if in_fence and fence_buf:
        code_block = "\n".join(fence_buf)
        if "{{" in code_block or "{%" in code_block:
            code_block = "{% raw %}\n" + code_block + "\n{% endraw %}"
        result_chunks.append(code_block)
    flush_prose()

    return "\n".join(result_chunks).strip() + "\n"


def parse_blog_readme_metadata(blog_readme_path: pathlib.Path) -> Dict[str, Dict[str, str]]:
    meta_by_stem: Dict[str, Dict[str, str]] = {}
    if not blog_readme_path.exists():
        return meta_by_stem
    text = blog_readme_path.read_text(encoding="utf-8")

    zh_badge_pat = re.compile(r"\*\*\[`\[([^\]]+)\]`[^\n]*?\]\(zh/([a-zA-Z0-9_-]+)\.md\)\*\*")
    for m in zh_badge_pat.finditer(text):
        badge_zh, stem = m.group(1).strip(), m.group(2).strip()
        meta_by_stem.setdefault(stem, {})["badge_zh"] = badge_zh
        after = text[m.end() :].lstrip(" \t")
        if after.startswith("\n") and after[1:].splitlines():
            next_line = after[1:].splitlines()[0].strip()
            if next_line.startswith("*") and next_line.endswith("*") and len(next_line) > 2:
                meta_by_stem[stem]["summary_zh"] = next_line.strip("*").strip()

    en_badge_pat = re.compile(r"\*\*\[`\[([^\]]+)\]`[^\n]*?\]\(en/([a-zA-Z0-9_-]+)\.md\)\*\*")
    for m in en_badge_pat.finditer(text):
        badge_en, stem = m.group(1).strip(), m.group(2).strip()
        meta_by_stem.setdefault(stem, {})["badge_en"] = badge_en
        after = text[m.end() :].lstrip(" \t")
        if after.startswith("\n") and after[1:].splitlines():
            next_line = after[1:].splitlines()[0].strip()
            if next_line.startswith("*") and next_line.endswith("*") and len(next_line) > 2:
                meta_by_stem[stem]["summary_en"] = next_line.strip("*").strip()

    return meta_by_stem


def split_blog_header_and_tldr(raw_md: str) -> Tuple[str, str, str]:
    _, body = strip_frontmatter(raw_md)
    lines = body.lstrip().splitlines()
    title = ""
    idx = 0
    if idx < len(lines) and lines[idx].startswith("# "):
        title = lines[idx][2:].strip()
        idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    tldr_lines: List[str] = []
    if idx < len(lines) and lines[idx].lstrip().startswith(">"):
        while idx < len(lines) and lines[idx].lstrip().startswith(">"):
            tldr_lines.append(re.sub(r"^\s*>\s?", "", lines[idx]))
            idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

    tldr_raw = " ".join(part.strip() for part in tldr_lines if part.strip())
    tldr_body = re.sub(r"^\*\*TL;DR\*\*\s*[:：]\s*", "", tldr_raw)
    tldr_body = re.sub(r"^\*\*TL;DR\s*[:：]\*\*\s*", "", tldr_body).strip()
    rest = "\n".join(lines[idx:]).strip()
    return title, tldr_body, rest


def summarize_first_sentence(tldr_text: str, max_chars: int = 160) -> str:
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", tldr_text)
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    clean = clean.replace("`", "")
    clean = re.sub(r"\$+([^$]+)\$+", r"\1", clean).strip()
    if len(clean) <= max_chars:
        return clean
    for sep in (". ", "。", "；", "; "):
        pos = clean.find(sep, 45)
        if 0 < pos <= max_chars:
            return clean[: pos + (1 if sep == "。" else 0)].rstrip("；;,，")
    cut = clean[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(",;；、:") + "..."


def extract_doc_summary(raw_md: str, is_english: bool) -> str:
    # First check if there is an explicit `> **本文解决**：...` or `> **What this solves**: ...` callout
    m_solves = re.search(r">\s*\*\*(?:本文解决|What this solves|场景|Scenario)\*\*\s*[:：]\s*([^\n]+)", raw_md)
    if m_solves:
        return summarize_first_sentence(m_solves.group(1), 145 if is_english else 95)

    para_lines: List[str] = []
    in_code = False
    for line in raw_md.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_code = not in_code
            if para_lines:
                break
            continue
        if in_code:
            continue
        if not s or s in ("---", "***", "___"):
            if para_lines:
                break
            continue
        if s.startswith(("#", "|", "<!--", "- ", "* ", ">", "![")):
            if para_lines:
                break
            continue
        para_lines.append(s)
    if not para_lines:
        return ""
    merged = (" " if is_english else "").join(para_lines)
    return summarize_first_sentence(merged, 145 if is_english else 95)


def parse_personal_readme_metadata(personal_readme_path: pathlib.Path) -> Dict[str, Dict[str, str]]:
    meta_by_stem: Dict[str, Dict[str, str]] = {}
    if not personal_readme_path.exists():
        return meta_by_stem
    text = personal_readme_path.read_text(encoding="utf-8")

    zh_pat = re.compile(r"\*\*\[`\[([^\]]+)\]`[^\n]*?\]\(([0-9a-zA-Z_-]+)\.md\)\*\*")
    for m in zh_pat.finditer(text):
        badge_zh, stem = m.group(1).strip(), m.group(2).strip()
        key = "personal-" + stem
        meta_by_stem.setdefault(key, {})["badge_zh"] = badge_zh
        after = text[m.end() :].lstrip(" \t")
        if after.startswith("\n") and after[1:].splitlines():
            next_line = after[1:].splitlines()[0].strip()
            if next_line.startswith("*") and next_line.endswith("*") and len(next_line) > 2:
                meta_by_stem[key]["summary_zh"] = next_line.strip("*").strip()

    en_pat = re.compile(r"\*\*\[`\[([^\]]+)\]`[^\n]*?\]\(en/([0-9a-zA-Z_-]+)\.md\)\*\*")
    for m in en_pat.finditer(text):
        badge_en, stem = m.group(1).strip(), m.group(2).strip()
        key = "personal-" + stem
        meta_by_stem.setdefault(key, {})["badge_en"] = badge_en

    return meta_by_stem


def sync_blogs(
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
    blog_dir = pllm_dir / "blog"
    zh_dir = blog_dir / "zh"
    en_dir = blog_dir / "en"
    personal_dir = blog_dir / "personal"
    personal_en_dir = personal_dir / "en"

    readme_meta = parse_blog_readme_metadata(blog_dir / "README.md")
    readme_meta.update(parse_personal_readme_metadata(personal_dir / "README.md"))

    zh_files = {p.stem: p for p in sorted(zh_dir.glob("*.md"))} if zh_dir.exists() else {}
    en_files = {p.stem: p for p in sorted(en_dir.glob("*.md"))} if en_dir.exists() else {}
    main_stems = sorted(set(zh_files.keys()) | set(en_files.keys()))

    p_zh_files = (
        {"personal-" + p.stem: p for p in sorted(personal_dir.glob("*.md")) if p.name.lower() != "readme.md"}
        if personal_dir.exists()
        else {}
    )
    p_en_files = (
        {"personal-" + p.stem: p for p in sorted(personal_en_dir.glob("*.md")) if p.name.lower() != "readme.md"}
        if personal_en_dir.exists()
        else {}
    )
    personal_stems = sorted(set(p_zh_files.keys()) | set(p_en_files.keys()))

    all_entries: List[Tuple[str, pathlib.Path, pathlib.Path, bool]] = []
    for s in main_stems:
        zf = zh_files.get(s) or en_files[s]
        ef = en_files.get(s) or zh_files[s]
        all_entries.append((s, zf, ef, False))
    for s in personal_stems:
        zf = p_zh_files.get(s) or p_en_files[s]
        ef = p_en_files.get(s) or p_zh_files[s]
        all_entries.append((s, zf, ef, True))

    raw_ts_by_stem: Dict[str, int] = {}
    for stem, zf, _, _ in all_entries:
        rel_check = zf.relative_to(pllm_dir).as_posix()
        ts_str = run_git(pllm_dir, ["log", "--diff-filter=A", "--follow", "--format=%at", "-1", "--", rel_check])
        raw_ts_by_stem[stem] = (
            int(ts_str) if ts_str.isdigit() else int(datetime.datetime(2026, 11, 23, 3, 0, tzinfo=CST).timestamp())
        )

    blog_url_by_stem: Dict[str, str] = {}
    dt_by_stem: Dict[str, datetime.datetime] = {}
    main_count = len(main_stems)
    personal_count = len(personal_stems)

    for idx, (stem, _, _, is_personal) in enumerate(all_entries, start=1):
        base_dt = datetime.datetime.fromtimestamp(raw_ts_by_stem[stem], tz=CST)
        raw_num_target = stem[len("personal-") :] if is_personal else stem
        m_num = re.match(r"^(\d+)-", raw_num_target)
        local_num = int(m_num.group(1)) if m_num else idx
        minute_offset = max(1, 30 - local_num) if not is_personal else max(1, 50 - local_num)
        post_dt = base_dt.replace(hour=base_dt.hour, minute=min(59, minute_offset), second=0, microsecond=0)
        dt_by_stem[stem] = post_dt
        blog_url_by_stem[stem] = "/blog/{}/pllm-{}/".format(post_dt.strftime("%Y"), stem)

    posts_dir = site_dir / "_posts"
    posts_dir.mkdir(parents=True, exist_ok=True)
    expected_filenames = set()
    blog_records: List[Dict[str, object]] = []

    arch_nums = {1, 2, 11}

    for idx, (stem, zh_path, en_path, is_personal) in enumerate(all_entries, start=1):
        zh_rel = zh_path.relative_to(pllm_dir).as_posix()
        en_rel = en_path.relative_to(pllm_dir).as_posix()

        title_zh, tldr_zh_raw, body_zh_raw = split_blog_header_and_tldr(zh_path.read_text(encoding="utf-8"))
        title_en, tldr_en_raw, body_en_raw = split_blog_header_and_tldr(en_path.read_text(encoding="utf-8"))

        tldr_zh = transform_markdown_document(
            tldr_zh_raw, zh_rel, pllm_dir, site_dir, anchor_map, blog_url_by_stem
        ).strip()
        tldr_en = transform_markdown_document(
            tldr_en_raw, en_rel, pllm_dir, site_dir, anchor_map, blog_url_by_stem
        ).strip()
        body_zh = transform_markdown_document(
            body_zh_raw, zh_rel, pllm_dir, site_dir, anchor_map, blog_url_by_stem
        ).strip()
        body_en = transform_markdown_document(
            body_en_raw, en_rel, pllm_dir, site_dir, anchor_map, blog_url_by_stem
        ).strip()

        rmeta = readme_meta.get(stem, {})
        badge_zh = rmeta.get("badge_zh", "排障手记" if is_personal else "技术拆解")
        badge_en = rmeta.get("badge_en", "Postmortem" if is_personal else "Deep Dive")
        summary_zh_base = (rmeta.get("summary_zh") or summarize_first_sentence(tldr_zh_raw, 130)).replace("`", "")
        summary_en_base = (rmeta.get("summary_en") or summarize_first_sentence(tldr_en_raw, 190)).replace("`", "")

        raw_num_target = stem[len("personal-") :] if is_personal else stem
        m_num = re.match(r"^(\d+)-", raw_num_target)
        local_num = int(m_num.group(1)) if m_num else idx

        if is_personal:
            track = "debugging-notes"
            order_num = 100 + local_num
            series_code = "N{:02d}".format(local_num)
            desc_zh = "{}（pLLM 底层排障手记 · 支持 EN / 中文 一键切换）".format(summary_zh_base.rstrip("。"))
            desc_en = "{} (pLLM Debugging Postmortem · Bilingual EN / 中文)".format(summary_en_base)
            post_tags = "pllm pytorch triton debugging"
            banner_zh = "> **pLLM 编译与底层排障手记（{}/{} · {}）** · [pLLM 项目主页与文档](/pllm/) · [专栏与排障索引](/pllm/#blog) · [GitHub 原文 ↗]({})".format(
                local_num, personal_count, badge_zh, "{}/blob/main/{}".format(REPO_WEB_BASE, zh_rel)
            )
            banner_en = "> **pLLM Debugging Postmortems ({}/{} · {})** · [pLLM Project & Docs](/pllm/) · [Series & Postmortems Index](/pllm/#blog) · [GitHub Source ↗]({})".format(
                local_num, personal_count, badge_en, "{}/blob/main/{}".format(REPO_WEB_BASE, en_rel)
            )
        else:
            track = "systems-architecture" if local_num in arch_nums else "engine-internals"
            order_num = local_num
            series_code = "#{:02d}".format(local_num)
            desc_zh = "{}（pLLM 技术专栏 · 支持 EN / 中文 一键切换）".format(summary_zh_base.rstrip("。"))
            desc_en = "{} (pLLM Engineering Series · Bilingual EN / 中文)".format(summary_en_base)
            post_tags = "pllm vllm inference-systems"
            banner_zh = "> **pLLM 技术随笔与机制拆解（{}/{} · {}）** · [pLLM 项目主页与文档](/pllm/) · [专栏总览与系统解剖图](/pllm/#blog) · [GitHub 原文 ↗]({})".format(
                local_num, main_count, badge_zh, "{}/blob/main/{}".format(REPO_WEB_BASE, zh_rel)
            )
            banner_en = "> **pLLM Engineering Essays & Deep Dives ({}/{} · {})** · [pLLM Project & Docs](/pllm/) · [Series Index & Anatomy Map](/pllm/#blog) · [GitHub Source ↗]({})".format(
                local_num, main_count, badge_en, "{}/blob/main/{}".format(REPO_WEB_BASE, en_rel)
            )

        post_dt = dt_by_stem[stem]
        filename = "{}-pllm-{}.md".format(post_dt.strftime("%Y-%m-%d"), stem)
        expected_filenames.add(filename)

        content_lines = [
            "---",
            "layout: post",
            "title: {}".format(yaml_quote(title_zh)),
            "title_en: {}".format(yaml_quote(title_en)),
            "date: {}".format(post_dt.strftime("%Y-%m-%d %H:%M:%S +0800")),
            "description: {}".format(yaml_quote(desc_zh)),
            "description_en: {}".format(yaml_quote(desc_en)),
            "tags: {}".format(post_tags),
            "categories: {}".format(track),
            "blog_track: {}".format(track),
            "pllm_synced: true",
            "pllm_track: {}".format(track),
            "pllm_order: {}".format(order_num),
            "pllm_code: {}".format(yaml_quote(series_code)),
            "pllm_badge_zh: {}".format(yaml_quote(badge_zh)),
            "pllm_badge_en: {}".format(yaml_quote(badge_en)),
            "related_posts: true",
            "---",
            "",
            '<div class="zine-dispatch" data-title-zh="{}" data-title-en="{}" data-desc-zh="{}" data-desc-en="{}">'.format(
                html_attr_escape(title_zh),
                html_attr_escape(title_en),
                html_attr_escape(desc_zh),
                html_attr_escape(desc_en),
            ),
            '  <a class="zine-pill-link" data-i18n-zh="pLLM 项目主页与文档 →" data-i18n-en="pLLM Project &amp; Docs →" href="/pllm/">pLLM 项目主页与文档 →</a>',
            '  <a class="zine-pill-link" data-i18n-zh="全部博客分类索引 →" data-i18n-en="Blog Categories →" href="/blog/">Blog Categories →</a>',
            "</div>",
            "",
            '<div class="zine-lang-panel" data-lang-panel="zh" lang="zh-CN" markdown="1">',
            "",
            '<div class="zine-tldr" markdown="1">',
            "",
            "**TL;DR：** {}".format(tldr_zh),
            "",
            "</div>",
            "",
            banner_zh,
            "",
            body_zh,
            "",
            "</div>",
            "",
            '<div class="zine-lang-panel is-hidden" data-lang-panel="en" lang="en" markdown="1">',
            "",
            '<div class="zine-tldr" markdown="1">',
            "",
            "**TL;DR:** {}".format(tldr_en),
            "",
            "</div>",
            "",
            banner_en,
            "",
            body_en,
            "",
            "</div>",
            "",
            '<script defer src="/assets/js/bilingual-post.js"></script>',
            "",
        ]
        (posts_dir / filename).write_text("\n".join(content_lines), encoding="utf-8")

        blog_records.append(
            {
                "order": order_num,
                "code": series_code,
                "track": track,
                "stem": stem,
                "url": blog_url_by_stem[stem],
                "title_zh": title_zh,
                "title_en": title_en,
                "badge_zh": badge_zh,
                "badge_en": badge_en,
                "summary_zh": summary_zh_base,
                "summary_en": summary_en_base,
            }
        )

    for existing in posts_dir.glob("*-pllm-*.md"):
        if existing.name not in expected_filenames:
            head = existing.read_text(encoding="utf-8")[:600]
            if "pllm_synced: true" in head:
                existing.unlink()

    blog_records.sort(key=lambda item: int(item["order"]))  # type: ignore[arg-type]
    return blog_url_by_stem, blog_records


def sync_docs(
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
    titles: Dict[str, str],
    blog_url_by_stem: Dict[str, str],
) -> Dict[str, List[Dict[str, str]]]:
    """Sync paired human-facing bilingual docs (`doc/` + `doc_en/`, excluding `doc/agent/`) into `_pllm_docs/`."""
    out_root = site_dir / "_pllm_docs"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sections: Dict[str, List[Dict[str, str]]] = {k: [] for k in DOC_CATEGORIES}
    doc_zh_root = pllm_dir / "doc"
    doc_en_root = pllm_dir / "doc_en"

    subpaths = set()
    for root_dir in (doc_zh_root, doc_en_root):
        if not root_dir.exists():
            continue
        for p in sorted(root_dir.rglob("*")):
            if not p.is_file() or ".ipynb_checkpoints" in p.parts:
                continue
            rel = p.relative_to(pllm_dir).as_posix()
            sub = normalize_doc_subpath(rel)
            if sub is not None:
                subpaths.add(sub)

    for sub in sorted(subpaths):
        zh_path = doc_zh_root / sub
        en_path = doc_en_root / sub
        has_zh = zh_path.exists()
        has_en = en_path.exists()
        if not has_zh and not has_en:
            continue

        zh_file = zh_path if has_zh else en_path
        en_file = en_path if has_en else zh_path
        zh_rel = zh_file.relative_to(pllm_dir).as_posix()
        en_rel = en_file.relative_to(pllm_dir).as_posix()

        is_notebook = pathlib.PurePosixPath(sub).suffix == ".ipynb"
        raw_zh = (
            notebook_to_raw_markdown(zh_file)
            if is_notebook
            else strip_frontmatter(zh_file.read_text(encoding="utf-8"))[1]
        )
        raw_en = (
            notebook_to_raw_markdown(en_file)
            if is_notebook
            else strip_frontmatter(en_file.read_text(encoding="utf-8"))[1]
        )

        title_zh = titles.get(zh_rel) or pathlib.PurePosixPath(sub).stem
        title_en = titles.get(en_rel) or title_zh

        body_zh = transform_markdown_document(
            raw_zh,
            zh_rel,
            pllm_dir,
            site_dir,
            anchor_map,
            blog_url_by_stem,
            attach_kramdown_ids=True,
            id_prefix="zh-",
            strip_leading_h1=True,
        ).strip()
        body_en = transform_markdown_document(
            raw_en,
            en_rel,
            pllm_dir,
            site_dir,
            anchor_map,
            blog_url_by_stem,
            attach_kramdown_ids=True,
            id_prefix="en-",
            strip_leading_h1=True,
        ).strip()

        summary_zh = extract_doc_summary(raw_zh, is_english=False)
        summary_en = extract_doc_summary(raw_en, is_english=True) or summary_zh

        parts = pathlib.PurePosixPath(sub).parts
        category = "root" if len(parts) == 1 else parts[0].lower()
        cat_zh, cat_en = DOC_CATEGORIES.get(category, ("文档总览", "Documentation Overview"))[:2]

        out_rel = doc_output_rel(sub)
        page_url = doc_url_for_rel("doc/" + sub) or "/pllm/docs/"
        gh_source_url = "{}/blob/main/doc/{}".format(REPO_WEB_BASE, sub)
        kind_badge_zh = "可执行教程 (.ipynb)" if is_notebook else cat_zh
        kind_badge_en = "Notebook (.ipynb)" if is_notebook else cat_en

        doc_lines = [
            "---",
            "layout: page",
            "title: {}".format(yaml_quote(title_en)),
            "description: {}".format(yaml_quote("pLLM Documentation · {}".format(cat_en))),
            "permalink: {}".format(page_url),
            "nav: false",
            "---",
            "",
            '<link rel="stylesheet" href="{{ \'/assets/css/pllm-hub.css\' | relative_url | bust_file_cache }}">',
            "",
            '<div class="zine-dispatch" data-title-zh="{}" data-title-en="{}" data-desc-zh="{}" data-desc-en="{}">'.format(
                html_attr_escape(title_zh),
                html_attr_escape(title_en),
                html_attr_escape("pLLM 文档 · {}".format(cat_zh)),
                html_attr_escape("pLLM Documentation · {}".format(cat_en)),
            ),
            '  <div class="zine-dispatch-links">',
            '    <a class="zine-pill-link primary" data-i18n-zh="← 返回 pLLM 文档中心" data-i18n-en="← pLLM Docs Hub" href="/pllm/#docs">← pLLM Docs Hub</a>',
            '    <a class="zine-pill-link" data-i18n-zh="文档导航地图" data-i18n-en="Docs Roadmap" href="/pllm/docs/">Docs Roadmap</a>',
            '    <span class="zine-lang-badge" data-i18n-zh="{}" data-i18n-en="{}">{}</span>'.format(
                html_attr_escape(kind_badge_zh),
                html_attr_escape(kind_badge_en),
                html_attr_escape(kind_badge_en),
            ),
            "  </div>",
            '  <a class="zine-pill-link" data-i18n-zh="{}" data-i18n-en="{}" href="{}" target="_blank" rel="noopener">{}</a>'.format(
                "在 GitHub 打开 .ipynb ↗" if is_notebook else "GitHub 源码 ↗",
                "Open .ipynb on GitHub ↗" if is_notebook else "View on GitHub ↗",
                gh_source_url,
                "Open .ipynb on GitHub ↗" if is_notebook else "View on GitHub ↗",
            ),
            "</div>",
            "",
            '<div id="markdown-content">',
            "",
            '<div class="zine-lang-panel" data-lang-panel="en" lang="en" markdown="1">',
            "",
            body_en,
            "",
            "</div>",
            "",
            '<div class="zine-lang-panel is-hidden" data-lang-panel="zh" lang="zh-CN" markdown="1">',
            "",
            body_zh,
            "",
            "</div>",
            "",
            "</div>",
            "",
            '<div class="zine-doc-footer">',
            '  <a class="zine-pill-link" data-i18n-zh="← 返回 pLLM 项目与文档中心" data-i18n-en="← Back to pLLM Project &amp; Docs" href="/pllm/#docs">← Back to pLLM Project &amp; Docs</a>',
            '  <a class="zine-pill-link" href="{}" target="_blank" rel="noopener">GitHub &middot; doc/{} ↗</a>'.format(
                gh_source_url, sub
            ),
            "</div>",
            "",
            '<script defer src="/assets/js/bilingual-post.js"></script>',
            "",
        ]

        dest_file = out_root / out_rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text("\n".join(doc_lines), encoding="utf-8")

        if category in DOC_CATEGORIES:
            sections[category].append(
                {
                    "title_zh": title_zh,
                    "title_en": title_en,
                    "url": page_url,
                    "github_url": gh_source_url,
                    "rel_path": "doc/" + sub,
                    "kind": "notebook" if is_notebook else "markdown",
                    "summary_zh": summary_zh,
                    "summary_en": summary_en,
                }
            )

    return sections


def write_sync_data_yaml(
    site_dir: pathlib.Path,
    commit_sha: str,
    full_sha: str,
    commit_date: str,
    doc_sections: Dict[str, List[Dict[str, str]]],
    blog_records: List[Dict[str, object]],
) -> None:
    lines: List[str] = [
        "# Auto-generated by bin/sync_pllm.py from LinesHogan/pLLM — do not edit manually.",
        "repo_url: {}".format(yaml_quote(REPO_WEB_BASE)),
        "commit_sha: {}".format(yaml_quote(commit_sha or "main")),
        "commit_url: {}".format(
            yaml_quote("{}/commit/{}".format(REPO_WEB_BASE, full_sha) if full_sha else REPO_WEB_BASE)
        ),
        "commit_date: {}".format(yaml_quote(commit_date)),
        "doc_categories:",
    ]

    ordered_cats = ["tutorials", "guides", "howto", "explain", "reference", "examples"]
    for cat in ordered_cats:
        items = doc_sections.get(cat, [])
        if not items:
            continue
        label_zh, label_en, short_zh, short_en = DOC_CATEGORIES[cat]
        lines.append("  - id: {}".format(yaml_quote(cat)))
        lines.append("    title_zh: {}".format(yaml_quote(label_zh)))
        lines.append("    title_en: {}".format(yaml_quote(label_en)))
        lines.append("    short_zh: {}".format(yaml_quote(short_zh)))
        lines.append("    short_en: {}".format(yaml_quote(short_en)))
        lines.append("    items:")
        for item in items:
            lines.append("      - title_zh: {}".format(yaml_quote(item["title_zh"])))
            lines.append("        title_en: {}".format(yaml_quote(item["title_en"])))
            lines.append("        url: {}".format(yaml_quote(item["url"])))
            lines.append("        github_url: {}".format(yaml_quote(item["github_url"])))
            lines.append("        rel_path: {}".format(yaml_quote(item["rel_path"])))
            lines.append("        kind: {}".format(yaml_quote(item["kind"])))
            lines.append("        summary_zh: {}".format(yaml_quote(item["summary_zh"])))
            lines.append("        summary_en: {}".format(yaml_quote(item["summary_en"])))

    lines.append("blogs:")
    for b in blog_records:
        lines.append("  - order: {}".format(b["order"]))
        lines.append("    code: {}".format(yaml_quote(str(b.get("code", "")))))
        lines.append("    track: {}".format(yaml_quote(str(b.get("track", "")))))
        lines.append("    stem: {}".format(yaml_quote(str(b["stem"]))))
        lines.append("    url: {}".format(yaml_quote(str(b["url"]))))
        lines.append("    title_zh: {}".format(yaml_quote(str(b["title_zh"]))))
        lines.append("    title_en: {}".format(yaml_quote(str(b["title_en"]))))
        lines.append("    badge_zh: {}".format(yaml_quote(str(b["badge_zh"]))))
        lines.append("    badge_en: {}".format(yaml_quote(str(b["badge_en"]))))
        lines.append("    summary_zh: {}".format(yaml_quote(str(b["summary_zh"]))))
        lines.append("    summary_en: {}".format(yaml_quote(str(b["summary_en"]))))

    out_file = site_dir / "_data" / "pllm_sync.yml"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pllm-dir", type=pathlib.Path, default=None, help="Local checkout of LinesHogan/pLLM")
    parser.add_argument(
        "--site-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Root directory of LinesHogan.github.io",
    )
    parser.add_argument("--repo-url", default=REPO_URL_DEFAULT, help="Git URL to clone when --pllm-dir is omitted")
    parser.add_argument("--branch", default="main", help="Git branch to clone when --pllm-dir is omitted")
    args = parser.parse_args()

    site_dir = args.site_dir.resolve()
    temp_dir: Optional[str] = None
    try:
        if args.pllm_dir is not None:
            pllm_dir = args.pllm_dir.resolve()
        else:
            temp_dir = tempfile.mkdtemp(prefix="pllm-sync-")
            pllm_dir = pathlib.Path(temp_dir) / "pLLM"
            subprocess.run(
                ["git", "clone", "--depth", "50", "--branch", args.branch, args.repo_url, str(pllm_dir)],
                check=True,
            )

        if not (pllm_dir / "README.md").exists():
            raise SystemExit("Error: {} does not look like a pLLM repository".format(pllm_dir))

        short_sha = run_git(pllm_dir, ["rev-parse", "--short", "HEAD"])
        full_sha = run_git(pllm_dir, ["rev-parse", "HEAD"])
        commit_date = run_git(pllm_dir, ["log", "-1", "--format=%cs"])

        copy_static_assets(pllm_dir, site_dir)
        anchor_map, titles = scan_doc_anchors(pllm_dir)
        blog_url_by_stem, blog_records = sync_blogs(pllm_dir, site_dir, anchor_map)
        doc_sections = sync_docs(pllm_dir, site_dir, anchor_map, titles, blog_url_by_stem)
        write_sync_data_yaml(site_dir, short_sha, full_sha, commit_date, doc_sections, blog_records)

        total_docs = sum(len(v) for v in doc_sections.values()) + 1  # +1 for index.md
        print(
            "Synced pLLM ({}) -> {} bilingual blog posts, {} human-only bilingual doc pages.".format(
                short_sha or "local", len(blog_records), total_docs
            )
        )
        return 0
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
