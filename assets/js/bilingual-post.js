(function () {
  function initBilingual() {
    var dispatch = document.querySelector(".zine-dispatch");
    var panels = document.querySelectorAll("[data-lang-panel]");
    var i18nNodes = document.querySelectorAll("[data-i18n-zh][data-i18n-en]");
    var placeholderNodes = document.querySelectorAll("[data-placeholder-zh][data-placeholder-en]");
    if (!panels.length && !i18nNodes.length && !placeholderNodes.length) return;

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

    var relatedBox = null;
    var relatedHeading = null;
    var relatedSubtitle = null;
    var relatedLinks = [];
    var postsI18nMap = null;
    var postsI18nRequested = false;

    document.querySelectorAll("h2").forEach(function (h2) {
      if (relatedBox) return;
      var txt = (h2.textContent || "").trim();
      if (txt === "Enjoy Reading This Article?") {
        relatedBox = h2.parentElement;
        relatedHeading = h2;
        if (h2.nextElementSibling && h2.nextElementSibling.tagName === "P") {
          relatedSubtitle = h2.nextElementSibling;
        }
        relatedLinks = relatedBox ? Array.prototype.slice.call(relatedBox.querySelectorAll("ul li a[href]")) : [];
      }
    });

    function normalizePath(href) {
      try {
        var u = new URL(href, window.location.origin);
        return u.pathname;
      } catch (e) {
        return href;
      }
    }

    function applyRelatedLang(target) {
      if (!relatedBox) return;
      if (relatedHeading) {
        relatedHeading.textContent = target === "zh" ? "喜欢这篇文章吗？" : "Enjoy Reading This Article?";
      }
      if (relatedSubtitle) {
        relatedSubtitle.textContent =
          target === "zh" ? "以下是几篇你可能也感兴趣的延伸阅读：" : "Here are some more articles you might like to read next:";
      }
      if (postsI18nMap && relatedLinks.length) {
        relatedLinks.forEach(function (a) {
          var key = normalizePath(a.getAttribute("href") || "");
          var entry = postsI18nMap[key] || postsI18nMap[key.replace(/\/$/, "")] || postsI18nMap[key + "/"];
          if (entry && entry[target]) {
            a.textContent = entry[target];
          }
        });
      }
      if (!postsI18nRequested && relatedLinks.length) {
        postsI18nRequested = true;
        fetch("/assets/json/posts-i18n.json")
          .then(function (r) {
            return r.ok ? r.json() : null;
          })
          .then(function (data) {
            if (!data) return;
            postsI18nMap = data;
            var current = document.documentElement.classList.contains("lang-zh") ? "zh" : "en";
            applyRelatedLang(current);
          })
          .catch(function () {});
      }
    }

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
      placeholderNodes.forEach(function (el) {
        el.setAttribute("placeholder", el.getAttribute(target === "en" ? "data-placeholder-en" : "data-placeholder-zh") || "");
      });
      document.documentElement.lang = target === "en" ? "en" : "zh-CN";
      document.documentElement.classList.toggle("lang-zh", target === "zh");
      document.documentElement.classList.toggle("lang-en", target === "en");
      applyRelatedLang(target);
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
