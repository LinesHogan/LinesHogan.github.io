(function () {
  function initBilingual() {
    var dispatch = document.querySelector(".zine-dispatch");
    var panels = document.querySelectorAll("[data-lang-panel]");
    var i18nNodes = document.querySelectorAll("[data-i18n-zh][data-i18n-en]");
    if (!panels.length && !i18nNodes.length) return;

    var menuList = document.querySelector("#navbar .navbar-menu-list");
    if (menuList && !menuList.querySelector(".zine-nav-lang")) {
      var navItem = document.createElement("li");
      navItem.className = "nav-item zine-nav-lang";
      var navSwitch = document.createElement("div");
      navSwitch.className = "zine-lang-switch";
      navSwitch.setAttribute("role", "group");
      navSwitch.setAttribute("aria-label", "Language");

      var btnEn = document.createElement("button");
      btnEn.type = "button";
      btnEn.className = "zine-lang-btn";
      btnEn.setAttribute("data-lang-target", "en");
      btnEn.setAttribute("aria-pressed", "false");
      btnEn.textContent = "EN";

      var btnZh = document.createElement("button");
      btnZh.type = "button";
      btnZh.className = "zine-lang-btn";
      btnZh.setAttribute("data-lang-target", "zh");
      btnZh.setAttribute("lang", "zh-CN");
      btnZh.setAttribute("aria-pressed", "false");
      btnZh.textContent = "中文";

      navSwitch.appendChild(btnEn);
      navSwitch.appendChild(btnZh);
      navItem.appendChild(navSwitch);

      var searchBtn = document.getElementById("search-toggle");
      var searchLi = searchBtn ? searchBtn.closest("li") : null;
      if (searchLi && searchLi.parentNode === menuList) {
        menuList.insertBefore(navItem, searchLi);
      } else {
        menuList.appendChild(navItem);
      }
    }

    var buttons = document.querySelectorAll("[data-lang-target]");
    var heading = document.querySelector(".post-header h1.post-title");
    var postDesc = document.querySelector(".post-header p.post-description");
    var titleSep = document.title.indexOf(" | ");
    var titleSuffix = titleSep !== -1 ? document.title.slice(titleSep) : "";

    function detectBrowserLang() {
      try {
        var langs =
          Array.isArray(navigator.languages) && navigator.languages.length
            ? navigator.languages
            : [navigator.language || navigator.userLanguage || ""];
        for (var i = 0; i < langs.length; i++) {
          var code = String(langs[i] || "").toLowerCase();
          if (!code) continue;
          if (code === "zh" || code.indexOf("zh-") === 0 || code.indexOf("zh_") === 0) {
            return "zh";
          }
          return "en";
        }
      } catch (e) {}
      return "en";
    }

    function resolveInitialLang() {
      try {
        var param = new URLSearchParams(window.location.search).get("lang");
        if (param === "en" || param === "zh") return param;
      } catch (e) {}
      try {
        var saved = window.localStorage.getItem("rh-lang");
        if (saved === "en" || saved === "zh") return saved;
      } catch (e) {}
      return detectBrowserLang();
    }

    function applyLang(lang, persist) {
      var target = lang === "zh" ? "zh" : "en";
      buttons.forEach(function (btn) {
        var active = btn.getAttribute("data-lang-target") === target;
        btn.classList.toggle("is-active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
      panels.forEach(function (panel) {
        var match = panel.getAttribute("data-lang-panel") === target;
        panel.classList.toggle("is-hidden", !match);
      });
      if (dispatch) {
        var nextTitle = dispatch.getAttribute(target === "en" ? "data-title-en" : "data-title-zh");
        if (nextTitle) {
          if (heading) heading.textContent = nextTitle;
          document.title = nextTitle + titleSuffix;
        }
        var nextDesc = dispatch.getAttribute(target === "en" ? "data-desc-en" : "data-desc-zh");
        if (nextDesc && postDesc) {
          postDesc.textContent = nextDesc;
        }
      }
      i18nNodes.forEach(function (el) {
        el.textContent = el.getAttribute(target === "en" ? "data-i18n-en" : "data-i18n-zh");
      });
      document.documentElement.lang = target === "en" ? "en" : "zh-CN";
      document.documentElement.classList.toggle("lang-zh", target === "zh");
      document.documentElement.classList.toggle("lang-en", target === "en");
      if (persist) {
        try {
          window.localStorage.setItem("rh-lang", target);
        } catch (e) {}
        try {
          var url = new URL(window.location.href);
          url.searchParams.set("lang", target);
          window.history.replaceState(null, "", url.toString());
        } catch (e) {}
      }
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        applyLang(btn.getAttribute("data-lang-target"), true);
      });
    });

    applyLang(resolveInitialLang(), false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initBilingual);
  } else {
    initBilingual();
  }
})();
