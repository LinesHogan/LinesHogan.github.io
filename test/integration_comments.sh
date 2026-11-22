#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_src="${tmp_dir}/src"
tmp_override="${tmp_dir}/comments-test-override.yml"
tmp_site="${tmp_dir}/site"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${tmp_src}"
tar -cf - --exclude=.git --exclude=_site --exclude=node_modules --exclude=resolution-hypothesis . | tar -xf - -C "${tmp_src}"

mkdir -p "${tmp_src}/_posts"
if [ ! -f "${tmp_src}/_posts/2022-12-10-giscus-comments.md" ]; then
  cat >"${tmp_src}/_posts/2022-12-10-giscus-comments.md" <<'MD'
---
layout: post
title: a post with giscus comments
date: 2022-12-10 11:59:00-0400
description: an example of a blog post with giscus comments
tags: comments
categories: sample-posts external-services
giscus_comments: true
related_posts: false
---

Fixture post for Giscus comments integration test.
MD
fi

if [ ! -f "${tmp_src}/_posts/2015-10-20-disqus-comments.md" ]; then
  cat >"${tmp_src}/_posts/2015-10-20-disqus-comments.md" <<'MD'
---
layout: post
title: a post with disqus comments
date: 2015-10-20 11:59:00-0400
description: an example of a blog post with disqus comments
tags: comments
categories: sample-posts external-services
disqus_comments: true
related_posts: false
---

Fixture post for Disqus comments integration test.
MD
fi

cat >"${tmp_override}" <<'YAML'
imagemagick:
  enabled: false
giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

bundle exec jekyll build --config "${tmp_src}/_config.yml,${tmp_override}" -s "${tmp_src}" -d "${tmp_site}" >/dev/null

giscus_page="${tmp_site}/blog/2022/giscus-comments/index.html"
disqus_page="${tmp_site}/blog/2015/disqus-comments/index.html"

grep -q 'https://giscus.app/client.js' "${giscus_page}"
if grep -q 'giscus comments misconfigured' "${giscus_page}"; then
  echo "unexpected giscus misconfiguration warning in ${giscus_page}" >&2
  exit 1
fi

grep -q 'id="disqus_thread"' "${disqus_page}"
grep -q '.disqus.com/embed.js' "${disqus_page}"

echo "comments integration checks passed"
