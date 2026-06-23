// Site-side override of hugo-ink's themes/hugo-ink/static/js/main.js.
// Hugo serves the project-root static/ over the theme's static/ at the same path,
// so this survives theme updates.
//
// Why it exists: the theme reads/writes localStorage unguarded AND assumes the
// feather global is always present. Brave (Shields up, "block all cookies", private
// window, or fingerprinting-protection edge cases) can throw a SecurityError on
// localStorage access, or make the feather global missing — either kills the whole
// DOMContentLoaded handler before the click listener gets wired up, so the scheme
// button does nothing while working in Chrome/Safari. This override guards every
// storage/matchMedia call and inlines the sun/moon SVGs so the toggle no longer
// depends on feather at runtime. Worst case (storage blocked): "works but does not
// persist across reloads" instead of "does not work at all".

var ICON_SUN  = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sun"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';
var ICON_MOON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-moon"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';

document.addEventListener("DOMContentLoaded", function () {
  var toggle = document.getElementById("scheme-toggle");
  if (!toggle) {
    // Toggle not in DOM (theme variant or stripped layout) — nothing to wire up.
    return;
  }

  var scheme = "light";

  var savedScheme = null;
  try {
    savedScheme = localStorage.getItem("scheme");
  } catch (e) {
    // localStorage blocked (Brave Shields / private mode) — fall back to system preference.
  }

  var container = document.getElementsByTagName("html")[0];

  var prefersDark = false;
  try {
    prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch (e) {
    // matchMedia unavailable or blocked — keep the light default.
  }

  if (prefersDark) {
    scheme = "dark";
  }

  if (savedScheme) {
    scheme = savedScheme;
  }

  if (scheme == "dark") {
    darkscheme(toggle, container);
  } else {
    lightscheme(toggle, container);
  }

  toggle.addEventListener("click", () => {
    if (toggle.className === "light") {
      darkscheme(toggle, container);
    } else if (toggle.className === "dark") {
      lightscheme(toggle, container);
    }
  });
});

function saveScheme(value) {
  try {
    localStorage.setItem("scheme", value);
  } catch (e) {
    // Storage blocked — the toggle still flips the theme, it just will not persist
    // across reloads. That is the best we can do when the browser refuses storage.
  }
}

function darkscheme(toggle, container) {
  saveScheme("dark");
  toggle.innerHTML = ICON_SUN;
  toggle.className = "dark";
  container.className = "dark";
}

function lightscheme(toggle, container) {
  saveScheme("light");
  toggle.innerHTML = ICON_MOON;
  toggle.className = "light";
  container.className = "";
}
