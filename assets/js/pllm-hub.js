(function () {
  function initPllmHub() {
    // 1. Capability / Code Showcase Tabs (/pllm/)
    var tabButtons = document.querySelectorAll("[data-pllm-tab]");
    var tabPanels = document.querySelectorAll("[data-pllm-panel]");
    if (tabButtons.length && tabPanels.length) {
      tabButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var target = btn.getAttribute("data-pllm-tab");
          tabButtons.forEach(function (b) {
            var active = b.getAttribute("data-pllm-tab") === target;
            b.classList.toggle("is-active", active);
            b.setAttribute("aria-selected", active ? "true" : "false");
          });
          tabPanels.forEach(function (panel) {
            var match = panel.getAttribute("data-pllm-panel") === target;
            panel.classList.toggle("is-hidden", !match);
          });
        });
      });
    }

    // 2. Documentation Category Filter Pills (/pllm/#docs)
    var filterButtons = document.querySelectorAll("[data-doc-filter]");
    var catBlocks = document.querySelectorAll("[data-doc-cat-block]");
    if (filterButtons.length && catBlocks.length) {
      filterButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var cat = btn.getAttribute("data-doc-filter");
          filterButtons.forEach(function (b) {
            var active = b.getAttribute("data-doc-filter") === cat;
            b.classList.toggle("is-active", active);
            b.setAttribute("aria-pressed", active ? "true" : "false");
          });
          catBlocks.forEach(function (block) {
            var match = cat === "all" || block.getAttribute("data-doc-cat-block") === cat;
            block.classList.toggle("is-hidden", !match);
          });
        });
      });
    }

    // 3. pLLM Blog Track Filter Pills (/pllm/#blog)
    var pllmBlogBtns = document.querySelectorAll("[data-pllm-blog-filter]");
    var pllmBlogBlocks = document.querySelectorAll("[data-pllm-blog-block]");
    if (pllmBlogBtns.length && pllmBlogBlocks.length) {
      pllmBlogBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var track = btn.getAttribute("data-pllm-blog-filter");
          pllmBlogBtns.forEach(function (b) {
            var active = b.getAttribute("data-pllm-blog-filter") === track;
            b.classList.toggle("is-active", active);
            b.setAttribute("aria-pressed", active ? "true" : "false");
          });
          pllmBlogBlocks.forEach(function (block) {
            var match = track === "all" || block.getAttribute("data-pllm-blog-block") === track;
            block.classList.toggle("is-hidden", !match);
          });
        });
      });
    }

    // 4. Main Blog Directory: Category Filter + View Mode + Instant Search (/blog/)
    var blogCatBtns = document.querySelectorAll("[data-blog-cat-btn]");
    var blogViewBtns = document.querySelectorAll("[data-blog-view-btn]");
    var blogSearch = document.querySelector("[data-blog-search]");
    var groupedContainer = document.getElementById("blog-view-grouped");
    var chronoContainer = document.getElementById("blog-view-chrono");
    var emptyNotice = document.getElementById("blog-empty-notice");

    if (blogCatBtns.length && (groupedContainer || chronoContainer)) {
      var activeCat = "all";
      var activeView = "grouped";
      var query = "";

      try {
        var urlCat = new URLSearchParams(window.location.search).get("cat");
        if (urlCat) {
          blogCatBtns.forEach(function (b) {
            if (b.getAttribute("data-blog-cat-btn") === urlCat) activeCat = urlCat;
          });
        }
      } catch (e) {}

      function applyBlogFilters() {
        var q = query.trim().toLowerCase();

        blogCatBtns.forEach(function (b) {
          var isMatch = b.getAttribute("data-blog-cat-btn") === activeCat;
          b.classList.toggle("is-active", isMatch);
          b.setAttribute("aria-pressed", isMatch ? "true" : "false");
        });

        blogViewBtns.forEach(function (b) {
          var isMatch = b.getAttribute("data-blog-view-btn") === activeView;
          b.classList.toggle("is-active", isMatch);
          b.setAttribute("aria-pressed", isMatch ? "true" : "false");
        });

        if (groupedContainer && chronoContainer) {
          groupedContainer.classList.toggle("is-hidden", activeView !== "grouped");
          chronoContainer.classList.toggle("is-hidden", activeView !== "chrono");
        }

        var visibleTotal = 0;
        var activeRoot = activeView === "grouped" ? groupedContainer : chronoContainer;
        if (!activeRoot) return;

        var items = activeRoot.querySelectorAll(".zine-blog-item");
        items.forEach(function (item) {
          var itemCat = item.getAttribute("data-post-cat") || "";
          var searchBlob = (item.getAttribute("data-post-search") || item.textContent || "").toLowerCase();
          var catOk = activeCat === "all" || itemCat === activeCat;
          var qOk = !q || searchBlob.indexOf(q) !== -1;
          var show = catOk && qOk;
          item.classList.toggle("is-hidden", !show);
          if (show) visibleTotal += 1;
        });

        if (activeView === "grouped" && groupedContainer) {
          var sections = groupedContainer.querySelectorAll(".zine-blog-section");
          sections.forEach(function (sec) {
            var secVisible = sec.querySelectorAll(".zine-blog-item:not(.is-hidden)").length;
            sec.classList.toggle("is-hidden", secVisible === 0);
          });
        }

        if (emptyNotice) {
          emptyNotice.classList.toggle("is-hidden", visibleTotal > 0);
        }
      }

      blogCatBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          activeCat = btn.getAttribute("data-blog-cat-btn") || "all";
          try {
            var url = new URL(window.location.href);
            if (activeCat === "all") {
              url.searchParams.delete("cat");
            } else {
              url.searchParams.set("cat", activeCat);
            }
            window.history.replaceState(null, "", url.toString());
          } catch (e) {}
          applyBlogFilters();
        });
      });

      blogViewBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          activeView = btn.getAttribute("data-blog-view-btn") || "grouped";
          applyBlogFilters();
        });
      });

      if (blogSearch) {
        blogSearch.addEventListener("input", function () {
          query = blogSearch.value || "";
          applyBlogFilters();
        });
      }

      applyBlogFilters();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPllmHub);
  } else {
    initPllmHub();
  }
})();
