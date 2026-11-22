(function () {
  function initPllmHub() {
    // 1. Capability / Code Showcase Tabs
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

    // 2. Documentation Category Filter Pills
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPllmHub);
  } else {
    initPllmHub();
  }
})();
