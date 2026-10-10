"""Take the README screenshots of the dashboard with headless Chrome (or Edge).

    hmis-dq dashboard --port 8502                  (another terminal; lake running)
    python docs/make_dashboard_screenshots.py --url http://localhost:8502

Drives the browser over the Chrome DevTools Protocol: opens each view through a
shareable link, waits until every chart is drawn, then saves a cropped PNG to
docs/images/. Uses a throwaway browser profile, never your own.
"""

import argparse
import base64
import itertools
import json
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import ClientConnection, connect

from hmis_dq.dashboard.data import open_dashboard_data
from hmis_dq.extract.catalog import DATASETS

BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
)
DEBUG_PORT = 9223
WIDTH = 1440
OUT = Path(__file__).parent / "images"

# The district shot: a district with a real early warning on show
DISTRICT, DATASET, INDICATOR = "Bo", DATASETS["reproductive_health"], "ANC 4th or more visits"


class Page:
    """Just enough of the DevTools Protocol: send a command, wait for its answer."""

    def __init__(self, ws_url: str) -> None:
        self.ws: ClientConnection = connect(ws_url, max_size=None)
        self.ids = itertools.count(1)

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        call_id = next(self.ids)
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params}))
        while True:  # skip events until our answer arrives
            message = json.loads(self.ws.recv())
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                result: dict[str, Any] = message.get("result", {})
                return result

    def js(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        return result["result"].get("value")

    def resize(self, height: int) -> None:
        self.call(
            "Emulation.setDeviceMetricsOverride",
            width=WIDTH,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )

    def open(self, url: str, charts: int, timeout: float = 90) -> None:
        """Load a view and wait until it has drawn ``charts`` charts and stopped running."""
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(1)
            drawn = self.js("document.querySelectorAll('.js-plotly-plot .main-svg').length")
            running = self.js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')")
            if (drawn or 0) >= charts and not running:
                time.sleep(2)  # let the last chart finish its transition
                return
        raise TimeoutError(f"{url} didn't finish drawing in {timeout:.0f}s")

    def fit_height(self) -> None:
        """Make the window as tall as the page, so nothing sits in a scroll area."""
        height = self.js("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
        self.resize(int(height) + 100)
        time.sleep(2)

    def save(self, path: Path, top: float, bottom: float, left: float = 0) -> None:
        clip = {"x": left, "y": top, "width": WIDTH - left, "height": bottom - top, "scale": 1}
        shot = self.call("Page.captureScreenshot", format="png", clip=clip)
        path.write_bytes(base64.b64decode(shot["data"]))
        print(f"saved {path} ({path.stat().st_size // 1024} KB)")


def find_browser() -> str:
    for candidate in BROWSERS:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    raise SystemExit("No Chrome or Edge found; install one or add it to BROWSERS.")


def start_browser(profile: str) -> tuple[subprocess.Popen[bytes], str]:
    browser = subprocess.Popen(
        [
            find_browser(),
            "--headless=new",
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--disable-extensions",
            "--hide-scrollbars",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list") as answer:
                targets = json.load(answer)
            page = next(t for t in targets if t["type"] == "page")
            return browser, str(page["webSocketDebuggerUrl"])
        except (OSError, StopIteration):
            time.sleep(0.5)
    browser.kill()
    raise SystemExit("The browser didn't start its DevTools endpoint.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501", help="where the dashboard runs")
    base = parser.parse_args().url.rstrip("/")

    districts = open_dashboard_data().districts(DATASET)
    district_id = districts.loc[districts["district"] == DISTRICT, "district_id"].iloc[0]
    link = urllib.parse.urlencode(
        {"dataset": DATASET, "district": district_id, "indicator": INDICATOR}
    )

    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="hmis-shots-")
    browser, ws_url = start_browser(profile)
    try:
        page = Page(ws_url)
        page.call(
            "Emulation.setEmulatedMedia",
            features=[{"name": "prefers-color-scheme", "value": "light"}],
        )
        page.resize(1000)

        # National overview (Child Health, the default view): title to the end of
        # the two overview charts, sidebar included
        page.open(f"{base}/", charts=5)
        page.fit_height()
        top = page.js("document.querySelector('h1').getBoundingClientRect().top")
        bottom = page.js(
            "Math.max(...[...document.querySelectorAll('.js-plotly-plot')].slice(0, 2)"
            ".map(p => p.getBoundingClientRect().bottom))"
        )
        page.save(OUT / "dashboard-overview.png", top - 30, bottom + 30)

        # District drill-down, from its header to the early-warning explanation
        page.resize(1000)
        page.open(f"{base}/?{link}", charts=5)
        page.fit_height()
        top = page.js("document.querySelector('h2').getBoundingClientRect().top")
        bottom = page.js(
            "Math.max(...[...document.querySelectorAll('[data-testid=\"stMarkdownContainer\"]')]"
            ".filter(e => e.textContent.includes('likely'))"
            ".map(e => e.getBoundingClientRect().bottom))"
        )
        left = page.js(
            "document.querySelector('[data-testid=\"stMain\"]').getBoundingClientRect().left"
        )
        page.save(OUT / "dashboard-district.png", top - 20, bottom + 30, left)
    finally:
        browser.kill()
        browser.wait()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
