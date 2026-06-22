// Site-side override of hugo-ink's themes/hugo-ink/static/js/main.js.
// Hugo serves the project-root static/ over the theme's static/ at the same path,
// so this survives theme updates.
//
// Why it exists: the theme reads/writes localStorage unguarded. Brave (Shields up,
// "block all cookies", or a private window) throws a SecurityError on any localStorage
// access, which killed the whole DOMContentLoaded handler before it could wire up the
// toggle — so the scheme button did nothing in Brave while working in Chrome/Safari.
// Guarding every storage and matchMedia call degrades that to "works but does not
// persist when storage is blocked" instead of "does not work at all".

document.addEventListener("DOMContentLoaded", function () {
  var toggle = document.getElementById("scheme-toggle");

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
  toggle.innerHTML = feather.icons.sun.toSvg();
  toggle.className = "dark";
  container.className = "dark";
}

function lightscheme(toggle, container) {
  saveScheme("light");
  toggle.innerHTML = feather.icons.moon.toSvg();
  toggle.className = "light";
  container.className = "";
}
