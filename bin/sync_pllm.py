#!/usr/bin/env python3
"""Synchronize blog posts and documentation from the pLLM repository into LinesHogan.github.io.

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

DOC_CATEGORY_LABELS = {
    "root": ("文档总览", "Documentation Index"),
    "agent": ("Agent Skill 与英文 API 规范", "Coding Agent Skill & English Reference"),
    "tutorials": ("交互式入门教程 (00–05)", "Step-by-Step Tutorials (00–05)"),
    "guides": ("用户指南", "User Guides"),
    "howto": ("操作与上线指南", "How-To & Serving Guides"),
    "explain": ("架构与机制解释", "Architecture & Design Explanations"),
    "reference": ("API 与配置参考手册", "API, Config & Support Reference"),
    "examples": ("算法族示例与实测 (06–13)", "Algorithm Family Examples (06–13)"),
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
    """Strip optional leading YAML frontmatter (`--- ... ---`) and return simple key/values + body."""
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
    """Compute a GitHub-compatible heading anchor slug (matching github-slugger)."""
    # Strip markdown links `[label](url)` -> `label`, inline code backticks, bold/italic markers
    clean = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", heading_text)
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    clean = clean.replace("`", "").replace("**", "").replace("*", "")
    clean = clean.strip().lower()
    # Remove punctuation characters that GitHub strips (keep word chars incl. Unicode letters/digits, spaces, hyphens)
    out_chars: List[str] = []
    for ch in clean:
        if ch == " " or ch == "-":
            out_chars.append("-")
        elif ch == "_" or ch.isalnum() or ord(ch) > 127:
            # Filter CJK/general punctuation
            if ch in "，。！？：；、“”‘’（）《》【】·—…—–/\\()[]{}:;,.!?\"'+~=<>&|$%@^#*":
                continue
            out_chars.append(ch)
    return "".join(out_chars).strip("-") or "section"


class SlugTracker:
    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.sec_counter = 0

    def register(self, raw_heading: str) -> Tuple[str, str]:
        base = github_slug(raw_heading)
        idx = self.counts.get(base, 0)
        self.counts[base] = idx + 1
        gh = base if idx == 0 else "{}-{}".format(base, idx)
        if re.match(r"^[a-z][a-z0-9_-]*$", gh):
            ascii_id = gh
        else:
            self.sec_counter += 1
            ascii_id = "sec-{}".format(self.sec_counter)
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


def doc_url_for_rel(rel_posix: str) -> Optional[str]:
    """Return site URL `/pllm/docs/.../` if `rel_posix` (relative to pLLM root) is a synced doc/notebook."""
    if not rel_posix.startswith("doc/"):
        return None
    sub = rel_posix[len("doc/") :]
    if sub.startswith("dev/"):
        return None
    if sub == "README.md":
        return "/pllm/docs/"
    p = pathlib.PurePosixPath(sub)
    if p.suffix in (".md", ".ipynb"):
        stem = p.stem.lower()
        parent = str(p.parent)
        if parent == ".":
            return "/pllm/docs/{}/".format(stem)
        return "/pllm/docs/{}/{}/".format(parent.lower(), stem)
    return None


def doc_output_rel(rel_posix: str) -> Optional[str]:
    """Return output path inside `_pllm_docs/` for a synced doc/notebook."""
    if not rel_posix.startswith("doc/"):
        return None
    sub = rel_posix[len("doc/") :]
    if sub.startswith("dev/"):
        return None
    if sub == "README.md":
        return "index.md"
    p = pathlib.PurePosixPath(sub)
    if p.suffix in (".md", ".ipynb"):
        stem = p.stem.lower()
        parent = str(p.parent)
        if parent == ".":
            return "{}.md".format(stem)
        return "{}/{}.md".format(parent.lower(), stem)
    return None


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


def scan_doc_anchors(pllm_dir: pathlib.Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Pre-scan all synced docs/notebooks to map `(rel_posix, github_slug) -> ascii_id` and extract H1 titles."""
    anchor_map: Dict[str, Dict[str, str]] = {}
    titles: Dict[str, str] = {}
    doc_root = pllm_dir / "doc"
    if not doc_root.exists():
        return anchor_map, titles

    for path in sorted(doc_root.rglob("*")):
        if not path.is_file() or ".ipynb_checkpoints" in path.parts:
            continue
        rel_posix = path.relative_to(pllm_dir).as_posix()
        if doc_url_for_rel(rel_posix) is None:
            continue
        raw = (
            notebook_to_raw_markdown(path)
            if path.suffix == ".ipynb"
            else strip_frontmatter(path.read_text(encoding="utf-8"))[1]
        )
        tracker = SlugTracker()
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

    if is_image and resolved.exists() and resolved.is_file():
        dest = site_dir / "assets" / "img" / "pllm" / rel_target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, dest)
        return "/assets/img/pllm/{}".format(rel_target)

    if rel_target in ("README.md", "README.zh.md", "README.en.md"):
        return "/pllm/"
    if rel_target == "blog/README.md":
        return "/pllm/#blog"
    if rel_target.startswith(("blog/zh/", "blog/en/")) and rel_target.endswith(".md"):
        stem = pathlib.PurePosixPath(rel_target).stem
        if stem in blog_url_by_stem:
            return blog_url_by_stem[stem]

    doc_url = doc_url_for_rel(rel_target)
    if doc_url is not None:
        if frag_part:
            tmap = anchor_map.get(rel_target, {})
            mapped_frag = tmap.get(frag_part.lower(), tmap.get(frag_part, frag_part))
            return "{}#{}".format(doc_url, mapped_frag)
        return doc_url

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
    strip_first_h1: bool = False,
) -> Tuple[str, bool]:
    """Transform non-fenced markdown prose for Kramdown + MathJax + site links."""
    chunk = chunk.replace("&nearr;", "↗")
    chunk = re.sub(r"\s*<!--\s*check-docs:[^>]*-->", "", chunk)

    # Protect inline code spans (`...`, ``...``)
    code_tokens: List[str] = []

    def save_code(m: re.Match) -> str:
        span = m.group(0)
        if "{{" in span or "{%" in span:
            span = "{% raw %}" + span + "{% endraw %}"
        idx = len(code_tokens)
        code_tokens.append(span)
        return "\x00CODE{}\x00".format(idx)

    chunk = re.sub(r"(`+)([^`\n]+?)\1", save_code, chunk)

    # Rewrite images `![alt](target)` and links `[text](target)`
    def replace_img(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2).strip()
        new_url = resolve_relative_url(
            url, source_rel_posix, pllm_dir, site_dir, anchor_map, blog_url_by_stem, is_image=True
        )
        return "![{}]({})".format(alt, new_url)

    def replace_link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2).strip()
        new_url = resolve_relative_url(
            url, source_rel_posix, pllm_dir, site_dir, anchor_map, blog_url_by_stem, is_image=False
        )
        return "[{}]({})".format(label, new_url)

    chunk = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", replace_img, chunk)
    chunk = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", replace_link, chunk)

    # Protect existing display/inline `$$...$$` math
    math_tokens: List[str] = []

    def save_double_math(m: re.Match) -> str:
        expr = m.group(1)
        idx = len(math_tokens)
        math_tokens.append((expr, True))
        return "\x00MATH{}\x00".format(idx)

    chunk = re.sub(r"\$\$(.+?)\$\$", save_double_math, chunk, flags=re.S)

    # Convert single-dollar inline math `$...$` -> `$$...$$`
    def save_single_math(m: re.Match) -> str:
        expr = m.group(1)
        idx = len(math_tokens)
        math_tokens.append((expr, False))
        return "\x00MATH{}\x00".format(idx)

    chunk = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", save_single_math, chunk)

    # Escape any stray Liquid tags outside code/math
    if "{{" in chunk or "{%" in chunk:
        chunk = chunk.replace("{{", "{% raw %}{{{% endraw %}").replace("{%", "{% raw %}{%{% endraw %}")

    # Restore math tokens (escaping bare `|` inside table rows so Kramdown tables never split cells)
    out_lines: List[str] = []
    h1_removed = strip_first_h1
    for line in chunk.splitlines():
        is_table_row = line.lstrip().startswith("|")

        def restore_line_math(m: re.Match) -> str:
            idx = int(m.group(1))
            expr, _ = math_tokens[idx]
            if is_table_row:
                # Replace single unescaped `|` (not `\|`) with `\vert ` inside GFM table cells
                expr = re.sub(r"(?<!\\)\|(?![|])", r"\\vert ", expr)
            if "{{" in expr or "{%" in expr:
                return "{% raw %}$$" + expr + "$${% endraw %}"
            return "$${}$$".format(expr)

        line = re.sub(r"\x00MATH(\d+)\x00", restore_line_math, line)
        line = re.sub(r"\x00CODE(\d+)\x00", lambda m: code_tokens[int(m.group(1))], line)
        out_lines.append(line)

    return "\n".join(out_lines), h1_removed


def transform_markdown_document(
    raw_text: str,
    source_rel_posix: str,
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
    blog_url_by_stem: Dict[str, str],
    attach_kramdown_ids: bool = False,
    strip_leading_h1: bool = False,
) -> str:
    """Transform a complete markdown document while preserving fenced code blocks."""
    lines = raw_text.splitlines()
    tracker = SlugTracker()
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
        converted, _ = convert_prose_chunk(
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
                # Remove internal check-docs skip directives from reader-facing code
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


def parse_blog_readme_metadata(
    blog_readme_path: pathlib.Path,
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Extract System Anatomy ASCII diagram, per-post badges, and per-post summaries from pLLM/blog/README.md."""
    meta_by_stem: Dict[str, Dict[str, str]] = {}
    anatomy_diagram = ""
    if not blog_readme_path.exists():
        return anatomy_diagram, meta_by_stem

    text = blog_readme_path.read_text(encoding="utf-8")
    diag_m = re.search(r"```text\n(.*?)```", text, flags=re.S)
    if diag_m:
        anatomy_diagram = diag_m.group(1).rstrip()

    # Parse ZH entries:
    zh_badge_pat = re.compile(r"\*\*\[`\[([^\]]+)\]`[^\n]*?\]\(zh/([a-zA-Z0-9_-]+)\.md\)\*\*")
    for m in zh_badge_pat.finditer(text):
        badge_zh, stem = m.group(1).strip(), m.group(2).strip()
        meta_by_stem.setdefault(stem, {})["badge_zh"] = badge_zh
        after = text[m.end() :].lstrip(" \t")
        if after.startswith("\n") and after[1:].splitlines():
            next_line = after[1:].splitlines()[0].strip()
            if next_line.startswith("*") and next_line.endswith("*") and len(next_line) > 2:
                meta_by_stem[stem]["summary_zh"] = next_line.strip("*").strip()

    # Parse EN entries:
    en_badge_pat = re.compile(r"\*\*\[`\[([^\]]+)\]`[^\n]*?\]\(en/([a-zA-Z0-9_-]+)\.md\)\*\*")
    for m in en_badge_pat.finditer(text):
        badge_en, stem = m.group(1).strip(), m.group(2).strip()
        meta_by_stem.setdefault(stem, {})["badge_en"] = badge_en
        after = text[m.end() :].lstrip(" \t")
        if after.startswith("\n") and after[1:].splitlines():
            next_line = after[1:].splitlines()[0].strip()
            if next_line.startswith("*") and next_line.endswith("*") and len(next_line) > 2:
                meta_by_stem[stem]["summary_en"] = next_line.strip("*").strip()

    return anatomy_diagram, meta_by_stem


def split_blog_header_and_tldr(raw_md: str) -> Tuple[str, str, str]:
    """Extract (title, single_paragraph_tldr_body, remaining_markdown) from a pLLM blog markdown file."""
    _, body = strip_frontmatter(raw_md)
    lines = body.lstrip().splitlines()
    title = ""
    idx = 0
    if idx < len(lines) and lines[idx].startswith("# "):
        title = lines[idx][2:].strip()
        # Strip surrounding Chinese book-title marks 《...》 only if wrapping the entire title? Keep original author title!
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


def summarize_first_sentence(tldr_text: str, max_chars: int = 180) -> str:
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", tldr_text)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"\$+([^$]+)\$+", r"\1", clean)
    if len(clean) <= max_chars:
        return clean
    # Cut cleanly at sentence boundary or word boundary
    for sep in (". ", "。", "；", "; "):
        pos = clean.find(sep, 60)
        if 0 < pos <= max_chars:
            return clean[: pos + (1 if sep in ("。", "；") else 1)].strip()
    cut = clean[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


def sync_blogs(
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
) -> Tuple[str, Dict[str, str], List[Dict[str, object]]]:
    """Sync paired ZH/EN pLLM blog posts into `_posts/*-pllm-*.md`."""
    blog_dir = pllm_dir / "blog"
    zh_dir = blog_dir / "zh"
    en_dir = blog_dir / "en"
    anatomy_diagram, readme_meta = parse_blog_readme_metadata(blog_dir / "README.md")

    zh_files = {p.stem: p for p in sorted(zh_dir.glob("*.md"))} if zh_dir.exists() else {}
    en_files = {p.stem: p for p in sorted(en_dir.glob("*.md"))} if en_dir.exists() else {}
    stems = sorted(set(zh_files.keys()) | set(en_files.keys()))

    # Determine first-commit dates per stem for stable URLs and ordering
    raw_ts_by_stem: Dict[str, int] = {}
    for stem in stems:
        rel_check = "blog/zh/{}.md".format(stem) if stem in zh_files else "blog/en/{}.md".format(stem)
        ts_str = run_git(pllm_dir, ["log", "--diff-filter=A", "--follow", "--format=%at", "-1", "--", rel_check])
        raw_ts_by_stem[stem] = int(ts_str) if ts_str.isdigit() else int(datetime.datetime(2026, 11, 23, 3, 0, tzinfo=CST).timestamp())

    # Precompute URLs `/blog/<year>/pllm-<stem>/`
    blog_url_by_stem: Dict[str, str] = {}
    dt_by_stem: Dict[str, datetime.datetime] = {}
    total_count = len(stems)
    for idx, stem in enumerate(stems, start=1):
        base_dt = datetime.datetime.fromtimestamp(raw_ts_by_stem[stem], tz=CST)
        m_num = re.match(r"^(\d+)-", stem)
        order_num = int(m_num.group(1)) if m_num else idx
        # Keep deterministic staggered timestamps on the commit date so #01 sits at top of the series on /blog/
        minute_offset = max(1, total_count + 1 - order_num)
        post_dt = base_dt.replace(
            hour=base_dt.hour,
            minute=min(59, minute_offset),
            second=0,
            microsecond=0,
        )
        dt_by_stem[stem] = post_dt
        blog_url_by_stem[stem] = "/blog/{}/pllm-{}/".format(post_dt.strftime("%Y"), stem)

    posts_dir = site_dir / "_posts"
    posts_dir.mkdir(parents=True, exist_ok=True)
    expected_filenames = set()
    blog_records: List[Dict[str, object]] = []

    for idx, stem in enumerate(stems, start=1):
        zh_path = zh_files.get(stem) or en_files[stem]
        en_path = en_files.get(stem) or zh_files[stem]
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
        badge_zh = rmeta.get("badge_zh", "技术拆解")
        badge_en = rmeta.get("badge_en", "Deep Dive")
        summary_zh_base = (rmeta.get("summary_zh") or summarize_first_sentence(tldr_zh_raw, 140)).replace("`", "")
        summary_en_base = (rmeta.get("summary_en") or summarize_first_sentence(tldr_en_raw, 200)).replace("`", "")
        desc_zh = "{}（pLLM 技术专栏 · 支持 EN / 中文 一键切换）".format(summary_zh_base.rstrip("。"))
        desc_en = "{} (pLLM Engineering Series · Bilingual EN / 中文)".format(summary_en_base)

        m_num = re.match(r"^(\d+)-", stem)
        order_num = int(m_num.group(1)) if m_num else idx
        post_dt = dt_by_stem[stem]
        filename = "{}-pllm-{}.md".format(post_dt.strftime("%Y-%m-%d"), stem)
        expected_filenames.add(filename)

        gh_zh_url = "{}/blob/main/{}".format(REPO_WEB_BASE, zh_rel)
        gh_en_url = "{}/blob/main/{}".format(REPO_WEB_BASE, en_rel)

        content_lines = [
            "---",
            "layout: post",
            "title: {}".format(yaml_quote(title_zh)),
            "title_en: {}".format(yaml_quote(title_en)),
            "date: {}".format(post_dt.strftime("%Y-%m-%d %H:%M:%S +0800")),
            "description: {}".format(yaml_quote(desc_zh)),
            "description_en: {}".format(yaml_quote(desc_en)),
            "tags: pllm vllm inference-systems",
            "categories: systems",
            "pllm_synced: true",
            "pllm_order: {}".format(order_num),
            "pllm_badge_zh: {}".format(yaml_quote(badge_zh)),
            "pllm_badge_en: {}".format(yaml_quote(badge_en)),
            "related_posts: true",
            "---",
            "",
            '<div class="zine-dispatch" data-title-zh="{}" data-title-en="{}">'.format(
                html_attr_escape(title_zh), html_attr_escape(title_en)
            ),
            '  <div class="zine-dispatch-links">',
            '    <a class="zine-pill-link" data-i18n-zh="pLLM 项目主页与文档" data-i18n-en="pLLM Project &amp; Docs" href="/pllm/">pLLM 项目主页与文档</a>',
            '    <a class="zine-pill-link" data-i18n-zh="GitHub · pLLM ↗" data-i18n-en="GitHub &middot; pLLM ↗" href="{}" target="_blank" rel="noopener">GitHub · pLLM ↗</a>'.format(
                REPO_WEB_BASE
            ),
            "  </div>",
            '  <div class="zine-lang-switch" role="group" aria-label="Language">',
            '    <button type="button" class="zine-lang-btn" data-lang-target="en" aria-pressed="false">EN</button>',
            '    <button type="button" class="zine-lang-btn is-active" data-lang-target="zh" lang="zh-CN" aria-pressed="true">中文</button>',
            "  </div>",
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
            "> **pLLM 技术随笔与机制拆解（{}/{} · {}）** · [pLLM 项目介绍与文档中心](/pllm/) · [专栏目录与系统解剖图](/pllm/#blog) · [仓库源码原文 ↗]({})".format(
                order_num, total_count, badge_zh, gh_zh_url
            ),
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
            "> **pLLM Engineering Essays & Deep Dives ({}/{} · {})** · [pLLM Project & Docs Hub](/pllm/) · [Series Index & Anatomy Map](/pllm/#blog) · [Markdown Source on GitHub ↗]({})".format(
                order_num, total_count, badge_en, gh_en_url
            ),
            "",
            body_en,
            "",
            "</div>",
            "",
            '<script defer src="/assets/js/bilingual-post.js"></script>',
            "",
        ]
        out_path = posts_dir / filename
        out_path.write_text("\n".join(content_lines), encoding="utf-8")

        blog_records.append(
            {
                "order": order_num,
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

    # Prune deleted synced pLLM posts without touching non-pLLM posts
    for existing in posts_dir.glob("*-pllm-*.md"):
        if existing.name not in expected_filenames:
            head = existing.read_text(encoding="utf-8")[:600]
            if "pllm_synced: true" in head:
                existing.unlink()

    blog_records.sort(key=lambda item: int(item["order"]))  # type: ignore[arg-type]
    return anatomy_diagram, blog_url_by_stem, blog_records


def sync_docs(
    pllm_dir: pathlib.Path,
    site_dir: pathlib.Path,
    anchor_map: Dict[str, Dict[str, str]],
    titles: Dict[str, str],
    blog_url_by_stem: Dict[str, str],
) -> Dict[str, List[Dict[str, str]]]:
    """Sync all pLLM docs (`doc/**/*.md` and `doc/{tutorials,examples}/*.ipynb`) into `_pllm_docs/`."""
    out_root = site_dir / "_pllm_docs"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sections: Dict[str, List[Dict[str, str]]] = {k: [] for k in DOC_CATEGORY_LABELS}
    doc_root = pllm_dir / "doc"
    if not doc_root.exists():
        return sections

    for path in sorted(doc_root.rglob("*")):
        if not path.is_file() or ".ipynb_checkpoints" in path.parts:
            continue
        rel_posix = path.relative_to(pllm_dir).as_posix()
        out_rel = doc_output_rel(rel_posix)
        page_url = doc_url_for_rel(rel_posix)
        if out_rel is None or page_url is None:
            continue

        sub_parts = pathlib.PurePosixPath(rel_posix[len("doc/") :]).parts
        category = "root" if len(sub_parts) == 1 else sub_parts[0].lower()
        cat_zh, cat_en = DOC_CATEGORY_LABELS.get(category, (category, category))
        is_notebook = path.suffix == ".ipynb"

        if is_notebook:
            raw_md = notebook_to_raw_markdown(path)
            fm_meta: Dict[str, str] = {}
        else:
            fm_meta, raw_md = strip_frontmatter(path.read_text(encoding="utf-8"))

        page_title = titles.get(rel_posix) or path.stem
        body_md = transform_markdown_document(
            raw_md,
            rel_posix,
            pllm_dir,
            site_dir,
            anchor_map,
            blog_url_by_stem,
            attach_kramdown_ids=True,
            strip_leading_h1=True,
        ).strip()

        # Extract short description from frontmatter or first non-heading paragraph
        short_desc = fm_meta.get("description", "")
        if not short_desc:
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
                if not s:
                    if para_lines:
                        break
                    continue
                if s.startswith(("#", "|", "<!--", "- ", "* ", ">")):
                    if para_lines:
                        break
                    continue
                para_lines.append(s)
            if para_lines:
                joiner = "" if category != "agent" else " "
                merged_para = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", joiner.join(para_lines))
                short_desc = summarize_first_sentence(merged_para, 150)

        gh_source_url = "{}/blob/main/{}".format(REPO_WEB_BASE, rel_posix)
        lang_tag = "EN" if category == "agent" else "中文"
        kind_badge = "Notebook (.ipynb)" if is_notebook else "Doc ({})".format(lang_tag)

        doc_lines = [
            "---",
            "layout: page",
            "title: {}".format(yaml_quote(page_title)),
            "description: {}".format(yaml_quote("pLLM · {} ({})".format(cat_en, cat_zh))),
            "permalink: {}".format(page_url),
            "nav: false",
            "---",
            "",
            '<div class="zine-dispatch">',
            '  <div class="zine-dispatch-links">',
            '    <a class="zine-pill-link primary" href="/pllm/#docs">← pLLM Project &amp; Docs</a>',
            '    <a class="zine-pill-link" href="/pllm/docs/">All Docs Index</a>',
            '    <span class="zine-lang-badge">{}</span>'.format(html_attr_escape(kind_badge)),
            "  </div>",
            '  <a class="zine-pill-link" href="{}" target="_blank" rel="noopener">{} ↗</a>'.format(
                gh_source_url,
                "Open .ipynb on GitHub" if is_notebook else "View Source on GitHub",
            ),
            "</div>",
            "",
            '<div id="markdown-content" markdown="1">',
            "",
            body_md,
            "",
            "</div>",
            "",
            '<div class="zine-doc-footer">',
            '  <a class="zine-pill-link" href="/pllm/#docs">← Back to pLLM Documentation Hub</a>',
            '  <a class="zine-pill-link" href="{}" target="_blank" rel="noopener">Edit / View on GitHub ↗</a>'.format(
                gh_source_url
            ),
            "</div>",
            "",
        ]

        dest_file = out_root / out_rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text("\n".join(doc_lines), encoding="utf-8")

        sections.setdefault(category, []).append(
            {
                "title": page_title,
                "url": page_url,
                "github_url": gh_source_url,
                "rel_path": rel_posix,
                "lang": lang_tag,
                "kind": "notebook" if is_notebook else "markdown",
                "summary": short_desc,
            }
        )

    return sections


def write_sync_data_yaml(
    site_dir: pathlib.Path,
    commit_sha: str,
    full_sha: str,
    commit_date: str,
    anatomy_diagram: str,
    doc_sections: Dict[str, List[Dict[str, str]]],
    blog_records: List[Dict[str, object]],
) -> None:
    """Emit `_data/pllm_sync.yml` for Liquid templates on `/pllm/`."""
    lines: List[str] = [
        "# Auto-generated by bin/sync_pllm.py from LinesHogan/pLLM — do not edit manually.",
        "repo_url: {}".format(yaml_quote(REPO_WEB_BASE)),
        "commit_sha: {}".format(yaml_quote(commit_sha or "main")),
        "commit_url: {}".format(
            yaml_quote("{}/commit/{}".format(REPO_WEB_BASE, full_sha) if full_sha else REPO_WEB_BASE)
        ),
        "commit_date: {}".format(yaml_quote(commit_date)),
    ]

    if anatomy_diagram:
        lines.append("anatomy_diagram: |2-")
        for dline in anatomy_diagram.splitlines():
            lines.append(("  " + dline) if dline else "")
    else:
        lines.append('anatomy_diagram: ""')

    lines.append("doc_categories:")
    ordered_cats = ["agent", "tutorials", "examples", "guides", "howto", "explain", "reference"]
    for cat in ordered_cats:
        items = doc_sections.get(cat, [])
        if not items:
            continue
        label_zh, label_en = DOC_CATEGORY_LABELS[cat]
        lines.append("  - id: {}".format(yaml_quote(cat)))
        lines.append("    title_zh: {}".format(yaml_quote(label_zh)))
        lines.append("    title_en: {}".format(yaml_quote(label_en)))
        lines.append("    items:")
        for item in items:
            lines.append("      - title: {}".format(yaml_quote(item["title"])))
            lines.append("        url: {}".format(yaml_quote(item["url"])))
            lines.append("        github_url: {}".format(yaml_quote(item["github_url"])))
            lines.append("        rel_path: {}".format(yaml_quote(item["rel_path"])))
            lines.append("        lang: {}".format(yaml_quote(item["lang"])))
            lines.append("        kind: {}".format(yaml_quote(item["kind"])))
            lines.append("        summary: {}".format(yaml_quote(item["summary"])))

    lines.append("blogs:")
    for b in blog_records:
        lines.append("  - order: {}".format(b["order"]))
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

        anchor_map, titles = scan_doc_anchors(pllm_dir)
        anatomy_diagram, blog_url_by_stem, blog_records = sync_blogs(pllm_dir, site_dir, anchor_map)
        doc_sections = sync_docs(pllm_dir, site_dir, anchor_map, titles, blog_url_by_stem)
        write_sync_data_yaml(site_dir, short_sha, full_sha, commit_date, anatomy_diagram, doc_sections, blog_records)

        total_docs = sum(len(v) for v in doc_sections.values())
        print(
            "Synced pLLM ({}) -> {} bilingual blog posts, {} doc/notebook pages.".format(
                short_sha or "local", len(blog_records), total_docs
            )
        )
        return 0
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
