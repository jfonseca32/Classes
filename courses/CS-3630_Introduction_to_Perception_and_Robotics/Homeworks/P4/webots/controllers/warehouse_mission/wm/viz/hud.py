"""The mission HUD: every supervisor label of the Webots layer.

This is the only module that calls ``Supervisor.setLabel``. The presentation
layer (``app.py``) tells the :class:`Hud` *what* to show (boot check,
briefing, countdown, live telemetry, announcements, score screen, ...) and
calls :meth:`Hud.tick` once per frame; the HUD animates (fades, typewriter,
blinking, count-ups) and re-sends only the labels that changed.

How Webots draws labels (measured on R2025a, see ``tools/qa/controllers/hud_check``):

* ids are 0-99 per robot (larger ids are rejected with a warning);
* every ``setLabel`` is a synchronous round trip to Webots (the C library
  flushes the request: ~0.5-1 ms), so only changed labels are sent and slow
  animations (pulses, the live score) change their labels a few times per
  second, not every frame;
* labels are drawn in *creation* order (the last created on top), so the HUD
  creates all 100 labels once, in id order, and a higher id is always on top
  (every drop shadow has a lower id than its text);
* ``size`` is twice the line height: one text line is ``0.5 * size`` of the
  3D view height; text widths follow from the font's advances (tables below)
  and the view's aspect ratio, which the HUD measures itself with a tiny
  ``exportImage`` (:meth:`Hud.measure_view`), so centering and right
  alignment are exact in any window shape;
* only artifact-free fonts are used: "Lucida Console" (monospace: telemetry,
  panels; different colors on one line are separate labels whose text is
  padded with spaces) and "Arial" (banners, drawn twice 1 px apart for a bold
  look). "Arial Black" and "Impact" show faint box outlines in Webots.

Layout (fractions of the 3D view, designed for 16:9, fine down to ~5:4): top-left
robot name, mission code and order; top-center phase banner; top-right mission
clock, score and stars; right column telemetry; bottom-center status toggles;
bottom-right ``[H] HELP`` and toasts. The bottom-left ~34% x 28% is left to the
minimap overlay. Full-screen panels (boot, briefing, score, fault, help) stay
above y = 0.71 or right of x = 0.35 for the same reason.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import textwrap
import time
from collections import deque
from collections.abc import Sequence

###############################################################################
#                                 Style                                       #
###############################################################################

CYAN = 0x19E6FF  # info / sensing
MAGENTA = 0xFF3DDC  # estimate / lost
YELLOW = 0xFFD21A  # goal / score
GREEN = 0x39FF88  # success
ORANGE = 0xFF8A00  # cargo / warnings
RED = 0xFF3B3B  # danger
WHITE = 0xE8F4FF  # values
LIME = 0xAAFF50  # the zone reading
DIM = 0xA4C2DA  # captions
FAINT = 0x5B7185  # rules, unearned stars
SHADOW = 0x02060C  # drop shadows

MONO = "Lucida Console"
SANS = "Arial"

# Glyph metrics at the 32 px size Webots rasterizes labels with: the FreeType line
# height (the label texture height of one line) and the integer advance of every
# printable ASCII character (32-126). Webots adds a margin of 0.15 line heights on
# each side of the text.
_ARIAL_ADVANCES = (
    9,
    9,
    11,
    18,
    18,
    28,
    21,
    6,
    11,
    11,
    12,
    19,
    9,
    11,
    9,
    9,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    9,
    9,
    19,
    19,
    19,
    18,
    32,
    21,
    21,
    23,
    23,
    21,
    20,
    25,
    23,
    9,
    16,
    21,
    18,
    27,
    23,
    25,
    21,
    25,
    23,
    21,
    20,
    23,
    21,
    30,
    21,
    21,
    20,
    9,
    9,
    9,
    15,
    18,
    11,
    18,
    18,
    16,
    18,
    18,
    9,
    18,
    18,
    7,
    7,
    16,
    7,
    27,
    18,
    18,
    18,
    18,
    11,
    16,
    9,
    18,
    16,
    23,
    16,
    16,
    16,
    11,
    8,
    11,
    19,
)
_FONTS = {SANS: (37, _ARIAL_ADVANCES), MONO: (32, (19,) * 95)}
_MARGIN = 0.15  # Webots' label margin, in line heights

# Label ids (Webots accepts 0-99). Within an element: shadow, text, bold pass.
_TOP_LEFT = range(0, 10)
_BANNER = range(10, 18)
_TOP_RIGHT = range(18, 28)
_TELEMETRY = range(28, 42)
_STATUS = range(42, 48)
_ANNOUNCE = range(48, 56)
_TOAST = range(56, 60)
_COUNTDOWN = range(60, 66)
_PANEL = range(66, 90)  # boot / briefing / scoreboard / fault (one at a time)
_HELP = range(90, 100)
_MAX_LABELS = 100

_HIDDEN = (" ", 0.0, 0.0, 0.01, 0, 1.0, SANS)  # what a label shows while unused
_ANNOUNCE_Y = (
    0.125  # announcements: a news line under the banner, the middle of the view stays free
)
_TELEMETRY_TOP = 0.36  # the telemetry column: right side, from here down to ~0.74

_KIND_COLORS = {"info": CYAN, "good": GREEN, "warn": ORANGE, "bad": RED, "big": YELLOW}
_STATUS_TAGS = {
    "OK": ("[ OK ]", GREEN),
    "MISSING": ("[MISSING]", RED),
    "ERROR": ("[ERROR]", ORANGE),
    "INFO": ("[INFO]", CYAN),
    "WARN": ("[WARN]", YELLOW),
}
_PHASE_COLORS = {
    "BOOT": CYAN,
    "BRIEFING": CYAN,
    "COUNTDOWN": CYAN,
    "LOCALIZING": CYAN,
    "TO_PICKUP": YELLOW,
    "DOCK_PICKUP": ORANGE,
    "TO_DOCK": ORANGE,
    "DOCK_DROPOFF": GREEN,
    "COMPLETE": GREEN,
    "FAILED": RED,
    "FAULT": RED,
}
_HELP_KEYS = (
    ("SPACE", "deploy (briefing)"),
    ("1-5 0", "cameras / free"),
    ("V", "robot vision"),
    ("B", "particles on/off"),
    ("M", "map layer"),
    ("K", "kidnap the robot"),
    ("O", "drop an obstacle"),
    ("F", "forklift traffic"),
    ("L", "night shift"),
    ("T", "turbo"),
    ("P", "photo"),
    ("S", "sound"),
    ("R", "retry mission"),
    ("N", "new mission"),
    ("E", "experiment (briefing)"),
    ("H", "close help"),
)
_CAMERAS = {
    "1": "CHASE",
    "2": "OVERHEAD",
    "3": "ROBOT POV",
    "4": "ORBIT",
    "5": "DIRECTOR",
    "0": "FREE",
}

# One label: (text, x, y, size, color, transparency, font).
Label = tuple[str, float, float, float, int, float, str]


def _ascii(text) -> str:
    """Webots labels only show ASCII 32-126 (plus newlines)."""
    return "".join(c if (32 <= ord(c) < 127 or c == "\n") else "?" for c in str(text))


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    """Word-wrap ``text`` into at most ``max_lines`` lines of ``width`` characters."""
    lines = textwrap.wrap(_ascii(text).replace("\n", " "), width, break_long_words=True) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 3)].rstrip() + "..."
    return lines


def _short_note(note, width: int = 26) -> str:
    """A persistent warning, shortened for the telemetry: "JACOBIAN FALLBACK: using ..." -> its
    head.
    """
    note = _ascii(note).strip()
    if len(note) > width:
        for sep in (": ", " ("):
            if sep in note:
                note = note.split(sep, 1)[0]
                break
    return note[:width]


def _num(value, default=None):
    """``value`` as a finite float, or ``default`` (None, NaN, inf, text, ...)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value, default: int = 0) -> int:
    number = _num(value)
    return default if number is None else int(round(number))


def _fmt_clock(seconds) -> str:
    seconds = min(max(0.0, _num(seconds, 0.0)), 99 * 60 + 59.9)
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - 60 * minutes:04.1f}"


def _steps(alpha: float, levels: int = 6) -> float:
    """``alpha`` rounded to a few levels: a slow pulse then changes its labels a few times per
    second instead of every frame (every changed label is a round trip to Webots)."""
    return round(alpha * levels) / levels


def _ease_out(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 3


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG header (the SOF marker), or None."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return (data[i + 7] << 8) | data[i + 8], (data[i + 5] << 8) | data[i + 6]
        if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
            i += 2 if marker != 0xFF else 1
            continue
        i += 2 + ((data[i + 2] << 8) | data[i + 3])
    return None


class Hud:
    """All the supervisor labels of the mission: build it once, call :meth:`tick` every frame.

    The panel methods (:meth:`boot`, :meth:`briefing`, :meth:`scoreboard`,
    :meth:`fault`) switch the screen; :meth:`telemetry` feeds the live mission
    HUD (shown once the countdown is over); :meth:`announce`, :meth:`toast`,
    :meth:`countdown` and :meth:`help` are overlays. Only :meth:`hide_all`,
    :meth:`restore` and :meth:`clear` send labels immediately; everything else
    is sent by the next :meth:`tick`.
    """

    def __init__(self, supervisor):
        self._sup = supervisor
        self._aspect = 16.0 / 9.0  # 3D view width / height; measured by measure_view()
        self._t = 0.0
        self._hidden = False
        self._sent: list[Label] = [_HIDDEN] * _MAX_LABELS
        self._errors: set = set()  # compose steps that failed once (reported on stderr)
        # QA statistics (the --autotest report): ticks, labels sent, view measurements, and the
        # seconds spent composing, sending (with and without the bookkeeping) the labels. Every
        # changed label is a synchronous round trip to Webots (~0.5-1 ms), hence the throttled
        # animations (_steps) and the few labels that change per frame.
        self.stats = {
            "ticks": 0,
            "labels": 0,
            "measures": 0,
            "compose_s": 0.0,
            "send_s": 0.0,
            "setlabel_s": 0.0,
        }
        self._reset_state()
        # Create every label once, in id order, so that higher ids are drawn on top.
        for i in range(_MAX_LABELS):
            self._send(i, _HIDDEN, force=True)
        self._measured = False
        self._measure_t = -10.0

    def _reset_state(self) -> None:
        self._screen: str | None = (
            None  # None | boot | briefing | countdown | mission | scoreboard | fault
        )
        self._screen_t = 0.0
        self._boot: tuple[str, list[tuple[str, str]], float] | None = None
        self._brief: dict = {}
        self._summary: dict = {}
        self._fault: dict = {}
        self._state: dict = {}
        self._count: tuple[str, float] | None = None
        self._queue: deque = deque()
        self._current: list | None = None  # [headline, subline, kind, duration, start, end]
        self._toast: tuple[str, float] | None = None
        self._help = False
        self._banner_text = ""
        self._banner_t = -10.0

    # =====================================================================
    # Public API (VIZ_SPEC 3.2)
    # =====================================================================
    def tick(self, dt: float) -> None:
        """Advance the animations by ``dt`` seconds and re-send the labels that changed."""
        self._t += max(0.0, _num(dt, 0.0))
        if not self._measured:
            self.measure_view()
        self._advance_announcements()
        self._flush()

    def boot(self, title: str, lines: list[tuple[str, str]], reveal: float) -> None:
        """SYSTEM CHECK screen: ``lines = [(text, status)]``, ``reveal`` = fraction of lines
        shown.
        """
        if self._screen != "boot":
            self._set_screen("boot")
        rows = []
        for line in lines or []:
            text, status = (
                (tuple(line) + ("", ""))[:2] if isinstance(line, (tuple, list)) else (line, "")
            )
            rows.append((_ascii(text if text is not None else ""), str(status or "").upper()))
        self._boot = (_ascii(title or ""), rows, min(1.0, max(0.0, _num(reveal, 0.0))))

    def briefing(self, info: dict) -> None:
        """Mission card: code, robot_name, student_name, orders, scenario(s), prompt, hints."""
        if self._screen != "briefing":
            self._set_screen("briefing")
        self._brief = dict(info or {})

    def countdown(self, n) -> None:
        """Big centered countdown: 3, 2, 1 (or any short text such as "GO"); ``None`` or "" clears
        it.
        """
        if n is None or n == "":
            self._count = None  # (queued announcements show right away)
            return
        if self._screen in ("boot", "briefing"):
            self._set_screen("countdown")
        self._count = (_ascii(n), self._t)

    def telemetry(self, state: dict) -> None:
        """Live mission HUD data (keys in VIZ_SPEC 3.2); cheap, call it every frame.

        The mission HUD replaces the countdown (or the briefing, once the state's
        phase is a mission phase); it never replaces the boot, score or fault screens.
        """
        self._state = state or {}
        deployed = str(self._state.get("phase", "")) not in ("", "BOOT", "BRIEFING", "COUNTDOWN")
        if self._screen in (None, "countdown") or (self._screen == "briefing" and deployed):
            self._set_screen("mission")

    def announce(self, text: str, kind: str = "info", duration: float = 2.5) -> None:
        """Queue a centered announcement (``kind``: info, good, warn, bad, big) that fades in and
        out.
        """
        if self._screen in ("boot", "briefing", "scoreboard", "fault"):
            return
        head, sub = self._split_announcement(_ascii(text).strip())
        if not head:
            return
        item = [head, sub, kind if kind in _KIND_COLORS else "info", max(0.8, _num(duration, 2.5))]
        if self._current is not None and self._current[0] == head and self._current[1] == sub:
            self._current[5] = max(
                self._current[5], self._t + item[3]
            )  # the same news again: keep it up
            return
        if any(q[0] == head and q[1] == sub for q in self._queue):
            return
        if kind == "big":
            self._queue.appendleft(item)
        else:
            self._queue.append(item)
        while (
            len(self._queue) > 4
        ):  # never fall far behind the action: drop the oldest ordinary news
            for k in range(len(self._queue) - 1, -1, -1):
                if self._queue[k][2] != "big":
                    del self._queue[k]
                    break
            else:
                self._queue.pop()

    def scoreboard(self, summary: dict) -> None:
        """MISSION COMPLETE / FAILED results screen (``mission.summary()`` plus extras)."""
        self._set_screen("scoreboard")
        self._summary = dict(summary or {})
        self._queue.clear()
        self._current = None

    def fault(self, fault: dict) -> None:
        """SYSTEM FAULT / STUDENT CODE ERROR / MISSING TODOs screen."""
        self._set_screen("fault")
        self._fault = dict(fault or {})
        self._queue.clear()
        self._current = None

    def help(self, visible: bool) -> None:
        """Show or hide the key help panel (``H``)."""
        visible = bool(visible)
        if visible and not self._help:
            self._maybe_measure()
        self._help = visible

    def toast(self, text: str) -> None:
        """Small bottom-right confirmation ("NIGHT SHIFT ON", "PHOTO SAVED"), fades after ~2 s (""
        clears it).
        """
        self._toast = (_ascii(text), self._t) if text else None

    def hide_all(self) -> None:
        """Photo mode: hide every label now (the state is kept for :meth:`restore`)."""
        self._hidden = True
        self._flush()

    def restore(self) -> None:
        """Show the HUD again after :meth:`hide_all`."""
        self._hidden = False
        self._flush()

    def clear(self) -> None:
        """Forget everything and remove every label now (restart)."""
        self._reset_state()
        self._flush()

    # --- extras -------------------------------------------------------------
    @property
    def aspect(self) -> float:
        """The 3D view's width / height used for the layout."""
        return self._aspect

    def measure_view(self) -> float | None:
        """Measure the 3D view's aspect ratio with a tiny synchronous screenshot (~20 ms).

        Called automatically at the first :meth:`tick` and when a full-screen
        panel or the help panel opens (at most once per second), so a resized
        window is picked up. The very first call can take ~0.15 s (Webots
        renders its first frame). Returns the aspect ratio, or None if Webots
        could not export an image (the previous value is kept).
        """
        self._measured = True
        self._measure_t = self._t
        self.stats["measures"] += 1
        path = os.path.join(tempfile.gettempdir(), f"wm_hud_view_{os.getpid()}.jpg")
        try:
            self._sup.exportImage(path, 1)
            with open(path, "rb") as handle:
                size = _jpeg_size(handle.read(1 << 16))
            os.remove(path)
        except Exception:  # noqa: BLE001  (no rendering, read-only temp dir, ...: keep the default)
            return None
        if not size or size[0] <= 0 or size[1] <= 0:
            return None
        self._aspect = min(3.5, max(0.8, size[0] / float(size[1])))
        return self._aspect

    # =====================================================================
    # Label bookkeeping
    # =====================================================================
    def _maybe_measure(self) -> None:
        """Re-measure the view (the window may have been resized), at most once per second."""
        if not self._measured or self._t - self._measure_t > 1.0:
            self.measure_view()

    def _set_screen(self, screen: str) -> None:
        self._screen = screen
        self._screen_t = self._t
        if screen in ("boot", "briefing", "scoreboard", "fault"):
            self._maybe_measure()

    def _send(self, i: int, label: Label, force: bool = False) -> None:
        if not force and self._sent[i] == label:
            return
        self.stats["labels"] += 1
        text, x, y, size, color, transparency, font = label
        try:
            t0 = time.perf_counter()
            self._sup.setLabel(
                i,
                text,
                min(1.0, max(0.0, x)),
                min(1.0, max(0.0, y)),
                size,
                int(color),
                min(1.0, max(0.0, transparency)),
                font,
            )
            self.stats["setlabel_s"] += time.perf_counter() - t0
        except Exception:  # noqa: BLE001  (a HUD glitch must never stop the mission)
            return
        self._sent[i] = label

    def _flush(self) -> None:
        start = time.perf_counter()
        want = {} if self._hidden else self._compose()
        composed = time.perf_counter()
        for i in range(_MAX_LABELS):
            self._send(i, want.get(i, _HIDDEN))
        stats = self.stats
        stats["ticks"] += 1
        stats["compose_s"] += composed - start
        stats["send_s"] += time.perf_counter() - composed

    # =====================================================================
    # Text geometry (fractions of the 3D view)
    # =====================================================================
    def text_width(self, text: str, size: float, font: str = MONO) -> float:
        """Width of a label (longest line, with Webots' margins) as a fraction of the view width."""
        vs, adv = _FONTS.get(font, _FONTS[SANS])
        widest = 0
        for line in str(text).split("\n"):
            widest = max(
                widest, sum(adv[ord(c) - 32] if 32 <= ord(c) <= 126 else adv[31] for c in line)
            )
        return 0.5 * size * (widest + 2 * _MARGIN * vs) / vs / self._aspect

    def _mono_w(self, chars: int, size: float) -> float:
        return 0.5 * size * (chars * 19 / 32 + 2 * _MARGIN) / self._aspect

    def _fit(self, text: str, size: float, font: str, max_w: float, min_size: float) -> float:
        """The largest size <= ``size`` (and >= ``min_size``) at which ``text`` fits in
        ``max_w``.
        """
        w = self.text_width(text, size, font)
        return size if w <= max_w else max(min_size, size * max_w / w)

    # =====================================================================
    # Label builders
    # =====================================================================
    def _shadow_offset(self, size: float) -> float:
        """Drop-shadow offset (fraction of the view height): ~5% of the text size, at most ~4 px."""
        return min(0.055 * size, 0.0055)

    def _text(
        self,
        out: dict[int, Label],
        ids,
        text: str,
        x: float,
        y: float,
        size: float,
        color: int,
        font: str = MONO,
        alpha: float = 1.0,
        shadow: float = 0.8,
        bold: bool = False,
    ) -> int:
        """One text element: a dark drop shadow, the text, and an optional faux-bold pass.

        Uses ids[0], ids[1], ... in order and returns how many ids it used.
        """
        if alpha <= 0.01 or not text:
            return 0
        k = 0
        off = self._shadow_offset(size)
        if shadow > 0:
            out[ids[k]] = (
                text,
                x + off / self._aspect,
                y + off,
                size,
                SHADOW,
                1.0 - shadow * alpha,
                font,
            )
            k += 1
        out[ids[k]] = (text, x, y, size, color, 1.0 - alpha, font)
        k += 1
        if bold:
            out[ids[k]] = (text, x + 0.016 * size / self._aspect, y, size, color, 1.0 - alpha, font)
            k += 1
        return k

    def _block(
        self,
        out: dict[int, Label],
        ids,
        rows: Sequence[Sequence[tuple[str, int]]],
        x: float,
        y: float,
        size: float,
        alpha: float = 1.0,
        shadow: float = 0.85,
        bold: Sequence[int] = (),
    ) -> None:
        """A multi-line, multi-color monospace block.

        ``rows`` is a list of rows, each a list of ``(text, color)``
        segments. Every color becomes one label whose other characters are
        spaces (Lucida Console is monospace, so the layers line up), plus one
        drop-shadow label with all the text. Colors listed in ``bold`` get a
        second pass a fraction of a pixel to the right (faux bold). Uses at
        most ``len(ids)`` ids; extra colors are drawn in the last layer's color.
        """
        if alpha <= 0.01:
            return
        layers: dict[int, list[str]] = {}
        order: list[int] = []
        plain: list[str] = []
        max_layers = max(1, len(ids) - (1 if shadow > 0 else 0) - len(bold))
        for r, segments in enumerate(rows):
            line = ""
            for text, color in segments:
                text = _ascii(text).replace("\n", " ")
                if not text:
                    continue
                if color not in layers:
                    if len(order) >= max_layers:
                        color = order[-1]
                    else:
                        layers[color] = [""] * len(rows)
                        order.append(color)
                layer = layers[color]
                layer[r] = layer[r].ljust(len(line)) + text
                line += text
            plain.append(line)
        k = 0
        if shadow > 0 and any(line.strip() for line in plain):
            off = self._shadow_offset(size)
            out[ids[k]] = (
                "\n".join(plain),
                x + off / self._aspect,
                y + off,
                size,
                SHADOW,
                1.0 - shadow * alpha,
                MONO,
            )
            k += 1
        for color in order:
            text = "\n".join(layers[color]).rstrip()
            if text.strip():
                out[ids[k]] = (text, x, y, size, color, 1.0 - alpha, MONO)
            k += 1
        for color in bold:
            text = "\n".join(layers.get(color, [])).rstrip()
            if text.strip() and k < len(ids):
                out[ids[k]] = (
                    text,
                    x + 0.02 * size / self._aspect,
                    y,
                    size,
                    color,
                    1.0 - alpha,
                    MONO,
                )
            k += 1

    # =====================================================================
    # Composition: what every label should show right now
    # =====================================================================
    def _compose(self) -> dict[int, Label]:
        """Every label the HUD should show now. A failing element is skipped (and reported once)."""
        out: dict[int, Label] = {}
        screen = self._screen
        parts = {
            "boot": [self._compose_boot],
            "briefing": [self._compose_briefing],
            "scoreboard": [self._compose_scoreboard],
            "fault": [self._compose_fault],
            "countdown": [self._compose_identity],
            "mission": [self._compose_identity, self._compose_mission],
        }.get(screen, [])
        parts = parts + [
            self._compose_countdown,
            self._compose_help if self._help else self._compose_announcement,
            self._compose_toast,
        ]
        for part in parts:
            try:
                part(out)
            except Exception as error:  # noqa: BLE001  (bad data must never stop the mission)
                name = part.__name__
                if name not in self._errors:
                    self._errors.add(name)
                    print(
                        f"wm.viz.hud: {name} skipped: {type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
        return out

    # --- mission HUD ----------------------------------------------------------
    def _compose_identity(self, out: dict[int, Label]) -> None:
        """Top-left: robot name, mission code, order and route."""
        s = self._state or {}
        brief = self._brief or {}
        name = _ascii(s.get("robot_name") or brief.get("robot_name") or "RB-3630")[:16]
        code = s.get("code", brief.get("code"))
        ids = list(_TOP_LEFT)
        x, y = 0.012, 0.012
        name_size = self._fit(name, 0.085, SANS, 0.25, 0.05)
        self._text(out, ids[0:3], name, x, y, name_size, WHITE, SANS, bold=True)
        rows = []
        if code is not None:
            rows.append([("MISSION ", DIM), (f"{_int(code) % 10000:04d}", YELLOW)])
        order = s.get("order")
        route = s.get("route")
        if order:
            row = [("ORDER   ", DIM), (_ascii(order), WHITE)]
            if route:
                row.append(("  " + _ascii(route)[:18], DIM))
            rows.append(row)
        self._block(out, ids[3:7], rows, x, y + 0.5 * name_size + 0.004, 0.046)

    def _compose_mission(self, out: dict[int, Label]) -> None:
        s = self._state
        t = self._t
        # ---- top-center: phase banner (types itself in when it changes) + status subline
        phase = str(s.get("phase", "") or "")
        relocalizing = bool(s.get("relocalizing"))
        banner = _ascii(s.get("banner") or phase.replace("_", " ")).upper()
        color = _PHASE_COLORS.get(phase, CYAN)
        if relocalizing and phase not in ("COMPLETE", "FAILED", "FAULT"):
            banner, color = "RE-LOCALIZING", MAGENTA
        if banner != self._banner_text:
            self._banner_text, self._banner_t = banner, t
        age = t - self._banner_t
        if banner:
            size = self._fit(banner, 0.105, SANS, self._center_width(), 0.055)
            shown = banner[: max(1, int(len(banner) * min(1.0, age / 0.35)))]
            x = 0.5 - self.text_width(banner, size, SANS) / 2
            flash = max(0.0, 1.0 - age / 0.45)
            col = WHITE if flash > 0.5 else color
            alpha = (
                _steps(0.55 + 0.45 * (0.5 + 0.5 * math.cos(2 * math.pi * 1.4 * t)))
                if relocalizing
                else 1.0
            )
            self._text(
                out, list(_BANNER)[:3], shown, x, 0.012, size, col, SANS, alpha=alpha, bold=True
            )
        sub, sub_color = self._banner_subline(s)
        if sub:
            size = self._fit(sub, 0.05, MONO, self._center_width(), 0.036)
            # Only the alarming LOST line pulses ("PARCEL ON BOARD" is news, not an alarm).
            blink = _steps(0.6 + 0.4 * (0.5 + 0.5 * math.cos(2 * math.pi * 1.2 * t)))
            self._text(
                out,
                list(_BANNER)[3:5],
                sub,
                0.5 - self._mono_w(len(sub), size) / 2,
                0.07,
                size,
                sub_color,
                MONO,
                alpha=blink if sub_color in (MAGENTA, RED) else 1.0,
            )

        # ---- top-right: mission clock, score and stars
        clock = _fmt_clock(s.get("clock", 0.0))
        size = 0.085
        x = 0.988 - self._mono_w(len(clock), size)
        self._text(out, list(_TOP_RIGHT)[:2], clock, x, 0.012, size, WHITE, MONO)
        if s.get("score") is not None:
            score = f"{max(0, min(99999, _int(s['score']))):5d} PTS"
            stars = max(0, min(3, _int(s.get("stars"))))
            star_text = " ".join("*" * 3)
            line = f"{score}  {star_text}"
            x = 0.988 - self._mono_w(len(line), 0.052)
            earned = " ".join("*" * stars)
            rows = [[(score + "  ", YELLOW), (earned, YELLOW), (star_text[len(earned) :], FAINT)]]
            self._block(out, list(_TOP_RIGHT)[2:6], rows, x, 0.062, 0.052)
        speed = _num(s.get("speed_factor"), 1.0)
        if speed >= 2.0:  # turbo: how much faster than real time the simulation runs
            text = f"x{min(speed, 999):.0f} SPEED"
            self._text(
                out,
                list(_TOP_RIGHT)[6:8],
                text,
                0.988 - self._mono_w(len(text), 0.045),
                0.098,
                0.045,
                ORANGE,
                MONO,
            )

        # ---- telemetry: the right column, or a compact block in the bottom band while the camera
        # shows
        # the whole warehouse from above (the building then fills the rest of the view); the help
        # panel takes its place while open
        if not self._help:
            if s.get("view") == "overhead":
                self._compose_telemetry_band(out, s)
            else:
                self._compose_telemetry(out, s)

        # ---- bottom: status toggles and the help hint
        self._compose_status(out, s)

    def _center_width(self) -> float:
        """Room for the centered banner between the top-left and top-right blocks."""
        s = self._state or {}
        name = _ascii(s.get("robot_name") or "RB-3630")[:16]
        left = 0.012 + max(
            self.text_width(name, self._fit(name, 0.085, SANS, 0.25, 0.05), SANS),
            self._mono_w(28, 0.046),
        )
        right = 0.012 + max(self._mono_w(7, 0.085), self._mono_w(15, 0.052))
        return max(0.25, 1.0 - 2.0 * max(left, right) - 0.03)

    @staticmethod
    def _banner_subline(s: dict) -> tuple[str, int]:
        if s.get("relocalizing"):
            return "LOST - SEARCHING AGAIN", MAGENTA
        if s.get("waiting"):
            return "WAITING FOR FORKLIFT", ORANGE
        phase = s.get("phase", "")
        parcel = s.get("parcel_state")
        if s.get("experiment"):  # a sensor test: which sensors are on
            return _ascii(s["experiment"]), CYAN
        if phase == "LOCALIZING":
            return "GLOBAL LOCALIZATION", CYAN
        if parcel == "robot" and phase in ("TO_DOCK", "DOCK_DROPOFF"):
            return "PARCEL ON BOARD", ORANGE
        return "", WHITE

    def _compose_telemetry(self, out: dict[int, Label], s: dict) -> None:
        """Right column: estimate +/- sigma, confidence bar, lock state, sensors, wheels, particles,
        notes.
        """
        t = self._t
        rows: list[list[tuple[str, int]]] = [[("TELEMETRY", CYAN)]]
        est = self._pair(s.get("estimate"))
        if est is not None:
            rows.append([("EST  ", DIM), (f"{est[0]:5.1f} {est[1]:5.1f}", WHITE)])
        sigma, error = _num(s.get("sigma")), _num(s.get("error"))
        if sigma is not None:
            row = [
                ("+/-  ", DIM),
                (f"{sigma:4.2f} m" if sigma < 10 else f"{min(sigma, 999.0):4.1f} m", WHITE),
            ]
            if error is not None:
                row.append((f"  err {min(error, 999.0):.2f}", FAINT))
            rows.append(row)
        conf = _num(s.get("confidence"))
        if conf is not None:
            c = min(1.0, max(0.0, conf))
            filled = int(round(10 * c))
            color = GREEN if c >= 0.8 else (YELLOW if c >= 0.4 else ORANGE)
            rows.append(
                [
                    ("[", DIM),
                    ("#" * filled, color),
                    ("-" * (10 - filled), FAINT),
                    ("]", DIM),
                    (f" {int(round(100 * c)):3d}%", color),
                ]
            )
        if s.get("relocalizing"):
            on = (t * 2.4) % 1.0 < 0.62
            rows.append([(">> LOST <<" if on else "", MAGENTA)])
        elif s.get("dead_reckoning"):
            rows.append([("DEAD RECKONING", YELLOW)])
        elif s.get("locked"):
            rows.append([("LOCKED", GREEN)])
        elif conf is not None:
            dots = "." * (1 + int(t * 3) % 3)
            rows.append([("LOCALIZING" + dots, CYAN)])
        rows.append([("", WHITE)])
        disabled = list(s.get("disabled") or [])
        if "z_on" in s:
            if "proximity" in disabled:
                rows.append([("PROX ", DIM), ("DISABLED", RED)])
            else:
                rows.append(
                    [
                        ("PROX ", DIM),
                        ("ON" if s.get("z_on") else "OFF", YELLOW if s.get("z_on") else WHITE),
                    ]
                )
        if "rfid" in s:
            rfid = self._pair(s.get("rfid"))
            if "rfid" in disabled:
                rows.append([("RFID ", DIM), ("DISABLED", RED)])
            elif rfid is not None and rfid[1] < 1e4:
                rows.append([("RFID ", DIM), (f"#{int(rfid[0])} @ {rfid[1]:.1f} m", ORANGE)])
            else:
                rows.append([("RFID ", DIM), ("out of range", FAINT)])
        if "zone" in s:
            zone = s.get("zone")
            if zone:
                rows.append([("ZONE ", DIM), (str(zone), LIME)])
            elif "zone" in disabled and s.get("experiment"):
                rows.append([("ZONE ", DIM), ("DISABLED", RED)])
            else:
                rows.append([("ZONE ", DIM), ("off", FAINT)])
        try:
            omegas = [_num(w) for w in list(s.get("wheels"))[:3]]
        except TypeError:
            omegas = []
        if len(omegas) == 3 and None not in omegas:  # the student's commanded wheel speeds (rad/s)
            text = " ".join(f"{max(-99.9, min(99.9, w)):+4.1f}" for w in omegas)
            rows.append([("W123 ", DIM), (text, WHITE)])
        if _num(s.get("particles")) is not None:
            rows.append([("N    ", DIM), (f"{_int(s['particles'])} particles", CYAN)])
        for note in list(s.get("notes") or [])[:3]:
            rows.append([("! " + _short_note(note), ORANGE)])
        size = 0.05
        width = self._mono_w(max(len("".join(seg for seg, _ in row)) for row in rows), size)
        x = 0.988 - max(width, self._mono_w(22, size))
        self._block(out, list(_TELEMETRY), rows, x, _TELEMETRY_TOP, size)

    def _compose_telemetry_band(self, out: dict[int, Label], s: dict) -> None:
        """The telemetry of the OVERHEAD view: a compact block right of the minimap, under the
        building.

        Rows: estimate +/- sigma (and the true error); confidence bar, lock state and particle
        count;
        proximity / RFID / zone readings; the warnings. Right-aligned at the view's right edge,
        from y = 0.78 (the camera frames the building above y = 0.75).
        """
        t = self._t
        rows: list[list[tuple[str, int]]] = []
        est = self._pair(s.get("estimate"))
        sigma, error = _num(s.get("sigma")), _num(s.get("error"))
        row: list[tuple[str, int]] = []
        if est is not None:
            row += [("EST ", DIM), (f"{est[0]:5.1f} {est[1]:5.1f}", WHITE)]
        if sigma is not None:
            row += [("  +/- ", DIM), (f"{min(sigma, 99.9):4.2f} m", WHITE)]
        if error is not None:
            row.append((f"  err {min(error, 99.9):.2f}", FAINT))
        if row:
            rows.append(row)
        conf = _num(s.get("confidence"))
        row = []
        if conf is not None:
            c = min(1.0, max(0.0, conf))
            filled = int(round(10 * c))
            color = GREEN if c >= 0.8 else (YELLOW if c >= 0.4 else ORANGE)
            row += [
                ("[", DIM),
                ("#" * filled, color),
                ("-" * (10 - filled), FAINT),
                ("]", DIM),
                (f" {int(round(100 * c)):3d}%  ", color),
            ]
        if s.get("relocalizing"):
            row.append(((">> LOST <<" if (t * 2.4) % 1.0 < 0.62 else "").ljust(13), MAGENTA))
        elif s.get("dead_reckoning"):
            row.append(("DEAD RECKONING ", YELLOW))
        elif s.get("locked"):
            row.append(("LOCKED".ljust(13), GREEN))
        elif conf is not None:
            row.append((("LOCALIZING" + "." * (1 + int(t * 3) % 3)).ljust(13), CYAN))
        if _num(s.get("particles")) is not None:
            row += [("N ", DIM), (f"{_int(s['particles'])}", CYAN)]
        if row:
            rows.append(row)
        row = []
        disabled = list(s.get("disabled") or [])
        if "z_on" in s:
            row.append(("PROX ", DIM))
            row.append(
                ("OFF!", RED)
                if "proximity" in disabled
                else ("ON " if s.get("z_on") else "OFF", YELLOW if s.get("z_on") else WHITE)
            )
        if "rfid" in s:
            rfid = self._pair(s.get("rfid"))
            row.append(("  RFID ", DIM))
            if "rfid" in disabled:
                row.append(("OFF!", RED))
            else:
                row.append(
                    (f"#{int(rfid[0])} @ {rfid[1]:4.1f} m", ORANGE)
                    if rfid is not None and rfid[1] < 1e4
                    else ("--", FAINT)
                )
        if "zone" in s:
            zone = s.get("zone")
            row.append(("  ZONE ", DIM))
            if zone:
                row.append((str(zone), LIME))
            else:
                row.append(
                    ("OFF!", RED) if "zone" in disabled and s.get("experiment") else ("off", FAINT)
                )
        if row:
            rows.append(row)
        notes = [_short_note(note) for note in list(s.get("notes") or [])[:2]]
        if notes:
            rows.append([("! " + "  ! ".join(notes), ORANGE)])
        if not rows:
            return
        size = 0.046
        width = self._mono_w(max(len("".join(seg for seg, _ in row)) for row in rows), size)
        self._block(out, list(_TELEMETRY), rows, 0.988 - width, 0.78, size)

    @staticmethod
    def _pair(value) -> tuple[float, float] | None:
        """Two finite numbers from a tuple/array, or None."""
        try:
            a, b = _num(value[0]), _num(value[1])
        except (TypeError, IndexError, KeyError):
            return None
        return None if a is None or b is None else (a, b)

    def _compose_status(self, out: dict[int, Label], s: dict) -> None:
        """Bottom-center toggles (fit to the space right of the minimap) and "[H] HELP"
        bottom-right.
        """
        cam = str(s.get("camera", "") or "")
        cam = _CAMERAS.get(cam, cam).upper()
        items = []
        if "vision" in s:
            items.append(("V", "VISION", "ON" if s.get("vision") else "OFF", bool(s.get("vision"))))
        if s.get("belief"):
            items.append(("B", "BELIEF", str(s["belief"]).upper(), True))
        if "map_layer" in s:
            layer = str(s.get("map_layer") or "none").upper()
            items.append(("M", "MAP", "OFF" if layer == "NONE" else layer, layer != "NONE"))
        if cam:
            items.append(("CAM", "CAM", cam, True))
        if "turbo" in s:
            items.append(("T", "TURBO", "ON" if s.get("turbo") else "OFF", bool(s.get("turbo"))))
        if "night" in s:
            items.append(("L", "NIGHT", "ON" if s.get("night") else "OFF", bool(s.get("night"))))
        if "forklifts" in s:
            items.append(
                ("F", "FORKLIFTS", "ON" if s.get("forklifts") else "OFF", bool(s.get("forklifts")))
            )
        if "muted" in s:
            items.append(("S", "SOUND", "OFF" if s.get("muted") else "ON", not s.get("muted")))
        size = 0.046
        x0, x1 = 0.355, 0.86
        help_text = "[H] HELP"
        self._text(
            out,
            list(_STATUS)[4:6],
            help_text,
            0.988 - self._mono_w(len(help_text), size),
            0.955,
            size,
            DIM,
            MONO,
        )
        if not items:
            return
        # Drop the words "VISION", "BELIEF", ... when space is short, then trailing items.
        for verbose in (True, False):
            segments: list[tuple[str, int]] = []
            for key, word, value, on in items:
                if segments:
                    segments.append(("  ", DIM))
                if verbose:
                    segments.append((f"[{key}] {word} " if key != "CAM" else "[1-5] CAM ", DIM))
                else:
                    segments.append((f"{key}:", DIM))
                segments.append((value, CYAN if on else FAINT))
            chars = sum(len(text) for text, _ in segments)
            if self._mono_w(chars, size) <= x1 - x0 or not verbose:
                break
        while self._mono_w(chars, size) > x1 - x0 and len(segments) > 3:
            segments = segments[:-3]
            chars = sum(len(text) for text, _ in segments)
        width = self._mono_w(chars, size)
        x = x0 + max(0.0, (x1 - x0 - width) / 2)
        self._block(out, list(_STATUS)[:4], [segments], x, 0.955, size)

    # --- overlays -----------------------------------------------------------------
    @staticmethod
    def _split_announcement(text: str) -> tuple[str, str]:
        """Headline and subline: split at an explicit newline, ": ", or a double space."""
        if "\n" in text:
            head, sub = text.split("\n", 1)
            return head.strip(), sub.strip().replace("\n", " ")
        for sep in (": ", "  "):
            if sep in text:
                head, sub = text.split(sep, 1)
                if head.strip():
                    return head.strip(), sub.strip()
        return text, ""

    def _advance_announcements(self) -> None:
        t = self._t
        if self._count is not None and self._current is None:
            return  # news waits until the countdown is over
        cur = self._current
        if cur is not None and self._queue and t - cur[4] > 0.9 and cur[5] > t + 0.35:
            cur[5] = t + 0.35  # news is waiting: wrap this one up quickly
        if cur is not None and t >= cur[5]:
            self._current = cur = None
        if cur is None and self._queue:
            head, sub, kind, duration = self._queue.popleft()
            self._current = [head, sub, kind, duration, t, t + duration]

    def _compose_announcement(self, out: dict[int, Label]) -> None:
        cur = self._current
        if cur is None:
            return
        head, sub, kind, _, start, end = cur
        t = self._t
        alpha = min(1.0, (t - start) / 0.15, max(0.0, (end - t) / 0.45))
        if alpha <= 0.01:
            return
        big = kind == "big"
        base = 0.15 if big else 0.115
        pop = 1.0 + 0.18 * max(0.0, 1.0 - (t - start) / 0.18)
        size = self._fit(head, base, SANS, 0.9, 0.06) * pop
        y = _ANNOUNCE_Y - 0.25 * (size - base)  # grow around the resting position
        x = 0.5 - self.text_width(head, size, SANS) / 2
        color = _KIND_COLORS[kind]
        ids = list(_ANNOUNCE)
        self._text(out, ids[:3], head, x, y, size, color, SANS, alpha=alpha, bold=True)
        if sub:
            sub_size = self._fit(sub, 0.06 if big else 0.055, MONO, 0.9, 0.04)
            sy = y + 0.5 * size * 1.02
            self._text(
                out,
                ids[3:5],
                sub,
                0.5 - self.text_width(sub, sub_size, MONO) / 2,
                sy,
                sub_size,
                WHITE,
                MONO,
                alpha=alpha,
            )

    def _compose_toast(self, out: dict[int, Label]) -> None:
        if self._toast is None:
            return
        text, start = self._toast
        t = self._t
        life = 2.4
        if t - start > life:
            self._toast = None
            return
        alpha = min(1.0, (t - start) / 0.12, (life - (t - start)) / 0.5)
        text = "> " + text
        size = 0.055
        slide = 0.01 * max(0.0, 1.0 - (t - start) / 0.15)
        self._text(
            out,
            list(_TOAST)[:2],
            text,
            0.988 - self._mono_w(len(text), size) + slide,
            0.905,
            size,
            GREEN,
            MONO,
            alpha=max(0.0, alpha),
        )

    def _compose_countdown(self, out: dict[int, Label]) -> None:
        """3, 2, 1 (big yellow digits that pop in and fade, under a "DEPLOYING IN" caption), then
        DEPLOY.
        """
        if self._count is None:
            return
        text, start = self._count
        e = self._t - start
        is_number = text.isdigit()
        life = 1.0 if is_number else 1.6
        if e > life:
            self._count = None
            return
        alpha = max(0.0, min(1.0, e / 0.06, (life - e) / 0.3))
        base = 0.56 if is_number else 0.2
        size = base * (1.0 + 0.45 * max(0.0, 1.0 - e / 0.2) ** 2)
        size = self._fit(text, size, SANS, 0.9, 0.05)
        color = YELLOW if is_number else GREEN
        x = 0.5 - self.text_width(text, size, SANS) / 2
        y = 0.40 - 0.25 * size
        ids = list(_COUNTDOWN)
        self._text(out, ids[:3], text, x, y, size, color, SANS, alpha=alpha, bold=True)
        if is_number:
            caption = "DEPLOYING IN"
            self._text(
                out,
                ids[3:5],
                caption,
                0.5 - self._mono_w(len(caption), 0.05) / 2,
                0.40 - 0.25 * base - 0.03,
                0.05,
                CYAN,
                MONO,
            )

    # --- panels -------------------------------------------------------------------
    def _panel_title(
        self,
        out: dict[int, Label],
        ids,
        text: str,
        y: float,
        size: float,
        color: int,
        alpha: float = 1.0,
        left: float | None = None,
    ) -> float:
        """A big Arial headline (centered unless ``left``); returns the y just below it."""
        size = self._fit(text, size, SANS, 0.9, 0.05)
        x = left if left is not None else 0.5 - self.text_width(text, size, SANS) / 2
        self._text(out, ids, text, x, y, size, color, SANS, alpha=alpha, bold=True)
        return y + 0.5 * size

    def _compose_boot(self, out: dict[int, Label]) -> None:
        """Power-on self test: typewriter list of checks with [ OK ] / [MISSING] tags."""
        if self._boot is None:
            return
        title, lines, reveal = self._boot
        ids = list(_PANEL)
        t = self._t
        x = 0.035
        y = self._panel_title(out, ids[0:3], "WAREHOUSE CONTROL SYSTEM", 0.05, 0.1, CYAN, left=x)
        head = [
            [("FLEET OS 4.0 // POWER-ON SELF TEST", DIM)],
            [(title[:58] if title else "CS 3630 AUTONOMOUS WAREHOUSE", WHITE)],
            [("-" * 58, FAINT)],
        ]
        self._block(out, ids[3:7], head, x, y + 0.005, 0.042)
        top = y + 0.005 + 3 * 0.021 + 0.008
        n = len(lines)
        shown = reveal * n
        size = min(0.05, max(0.03, (0.70 - top) / max(n + 1, 1) / 0.5))
        text_cols = 36
        rows: list[list[tuple[str, int]]] = []
        for k, (text, status) in enumerate(lines):
            if k > shown:
                break
            text = text[: text_cols - 2]
            if k + 1 <= shown:  # complete line: dot leader + status tag
                tag, color = _STATUS_TAGS.get(status, (f"[{status[:7]}]", WHITE))
                leader = " " + "." * max(2, text_cols - len(text) - 1) + " "
                rows.append([(text, WHITE), (leader, FAINT), (tag, color)])
            else:  # the line being typed, with a blinking cursor
                typed = text[: int(len(text) * (shown - k))]
                cursor = "_" if (t * 2.5) % 1.0 < 0.6 else ""
                rows.append([(typed, WHITE), (cursor, CYAN)])
        if shown >= n:
            ok = all(status in ("OK", "INFO", "WARN") for _, status in lines)
            blocked = any(
                text.upper().startswith("CANNOT") for text, _ in lines
            )  # e.g. "CANNOT DEPLOY: ..."
            if ok:
                done, color = "ALL SYSTEMS NOMINAL", GREEN
            elif blocked:
                done, color = "CHECK FAILED - SEE [MISSING] / [ERROR] ABOVE", ORANGE
            else:  # the robot deploys, but something is off
                done, color = "DEGRADED MODE - SEE [ERROR] / [MISSING] ABOVE", YELLOW
            cursor = " _" if (t * 2.5) % 1.0 < 0.6 else ""
            rows.append([(">> " + done, color), (cursor, CYAN)])
        self._block(out, ids[7:16], rows, x, top, size, bold=(WHITE,))

    def _compose_briefing(self, out: dict[int, Label]) -> None:
        """Mission card: code, robot, operator, orders, hints, and the blinking deploy prompt."""
        info = self._brief
        ids = list(_PANEL)
        t = self._t - self._screen_t
        alpha = min(1.0, t / 0.3)
        y = self._panel_title(out, ids[0:3], "MISSION BRIEFING", 0.07, 0.12, CYAN, alpha=alpha)
        code = info.get("code")
        cols = 50
        rows: list[list[tuple[str, int]]] = []
        if code is not None:
            rows.append([("MISSION CODE  ", DIM), (f"{_int(code) % 10000:04d}", YELLOW)])
        if info.get("robot_name"):
            rows.append([("ROBOT         ", DIM), (_ascii(info["robot_name"])[:24], WHITE)])
        if info.get("student_name"):
            rows.append([("OPERATOR      ", DIM), (_ascii(info["student_name"])[:34], WHITE)])
        rows.append([("-" * cols, FAINT)])
        scenarios = [tuple(item) for item in list(info.get("scenarios") or []) if len(item) == 2]
        if scenarios:
            rows.append(
                [
                    ("EXPERIMENT    ", DIM),
                    (_ascii(info.get("scenario_title") or "MISSION")[:34], CYAN),
                ]
            )
        orders = [o for o in list(info.get("orders") or []) if isinstance(o, dict)]
        if info.get("description") and (
            not orders or info.get("options")
        ):  # an experiment: what it does
            for line in list(info["description"])[:3]:
                for part in _wrap(line, cols - 2, 2):
                    rows.append([("  " + part, WHITE)])
        options = [_ascii(o) for o in list(info.get("options") or [])][:9]
        if options:  # an experiment: keys 1-N pick its test; the current one in green
            current = _ascii(info.get("option") or "")
            head = f"{_ascii(info.get('option_title') or 'PICK')[:10]} [1-{len(options)}]"
            lead = max(14, len(head) + 1)
            row: list[tuple[str, int]] = [(f"{head:<{lead}}", DIM)]
            used = lead
            for n, option in enumerate(options, start=1):
                cell = f"{n} {option}"
                if used > lead and used + 2 + len(cell) > cols:  # wrap under the first option
                    rows.append(row)
                    row, used = [(" " * lead, DIM)], lead
                gap = "  " if used > lead else ""
                row += [(gap + f"{n} ", YELLOW), (option, GREEN if option == current else FAINT)]
                used += len(gap) + len(cell)
            rows.append(row)
        for k, order in enumerate(orders[:6]):
            number = _int(order.get("number"), k + 1)
            rows.append(
                [
                    (f"ORDER {number}   ", DIM),
                    ("PICK UP ", DIM),
                    (f"{_ascii(order.get('pickup', '?')):<7}", YELLOW),
                    (" >> ", FAINT),
                    ("DELIVER TO ", DIM),
                    (_ascii(order.get("dock", "?")), GREEN),
                ]
            )
        if orders or info.get("description"):
            rows.append([("-" * cols, FAINT)])
        for hint in list(info.get("hints") or [])[:5]:
            for k, line in enumerate(_wrap(hint, cols - 2, 2)):
                rows.append([("> " if k == 0 else "  ", CYAN), (line, WHITE)])
        if (
            len(scenarios) > 1
        ):  # E cycles the experiments; the current one in green (wrapped to the card)
            rows.append([("", WHITE)])
            row, used = [("[E] ", YELLOW)], 4
            for key, title in scenarios:
                name = _ascii(title)
                if used > 4 and used + 1 + len(name) > cols:
                    rows.append(row)
                    row, used = [("    ", YELLOW)], 4
                row.append(
                    (
                        (" " if used > 4 else "") + name,
                        GREEN if key == info.get("scenario") else FAINT,
                    )
                )
                used += (1 if used > 4 else 0) + len(name)
            rows.append(row)
        size = self._fit("M" * cols, 0.056, MONO, 0.96, 0.04)
        x = max(0.01, 0.5 - self._mono_w(cols, size) / 2)
        self._block(out, ids[3:10], rows, x, y + 0.02, size, alpha=alpha)
        bottom = y + 0.02 + 0.5 * size * len(rows)
        prompt = _ascii(info.get("prompt") or "PRESS SPACE TO DEPLOY")
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.cos(2 * math.pi * 1.1 * t))
        text = f">>  {prompt}  <<"
        psize = self._fit(text, 0.075, SANS, 0.62, 0.045)
        py = min(0.66, bottom + 0.025)
        self._text(
            out,
            ids[10:13],
            text,
            0.5 - self.text_width(text, psize, SANS) / 2,
            py,
            psize,
            YELLOW,
            SANS,
            alpha=alpha * pulse,
            bold=True,
        )

    def _compose_scoreboard(self, out: dict[int, Label]) -> None:
        """Game results screen: headline, stars and rank on top, the stats table on the left, the
        score
        (counting up) and the retry prompt on the right; the middle stays free for the robot.
        """
        sm = self._summary
        if sm.get("experiment"):
            self._compose_experiment_board(out)
            return
        ids = list(_PANEL)
        e = self._t - self._screen_t
        success = bool(sm.get("success"))
        failed_text = "MISSION FAILED" if sm.get("phase") != "FAULT" else "MISSION ABORTED"
        head = _ascii(
            sm.get("headline") or ("MISSION COMPLETE" if success else failed_text)
        ).upper()
        y = self._panel_title(
            out, ids[0:3], head, 0.035, 0.13, GREEN if success else RED, alpha=min(1.0, e / 0.25)
        )
        # Stars pop in one by one, then the rank title.
        stars = max(0, min(3, _int(sm.get("stars"))))
        star_size = 0.3
        slots = "*   *   *"
        x = 0.5 - self.text_width(slots, star_size, SANS) / 2
        # The asterisk glyph only fills 16%..41% of its line box: place the box so the stars sit
        # just below the headline, and continue below the glyphs (not below the box).
        y -= 0.5 * star_size * 0.16 - 0.012
        shown = sum(1 for k in range(stars) if e > 0.9 + 0.35 * k)
        earned = "   ".join("*" * shown)
        if (
            e > 0.6
        ):  # shadow, the empty slots, the earned stars (Arial is not monospace: keep the spacing)
            off = self._shadow_offset(star_size)
            out[ids[3]] = (slots, x + off / self._aspect, y + off, star_size, SHADOW, 0.4, SANS)
            out[ids[4]] = (slots, x, y, star_size, FAINT, 0.0, SANS)
            if earned:
                out[ids[5]] = (earned, x, y, star_size, YELLOW, 0.0, SANS)
        y += 0.5 * star_size * 0.41 + 0.012
        title = _ascii(sm.get("title") or "")
        if title and e > 0.9 + 0.35 * max(stars, 1):
            self._panel_title(out, ids[6:9], title, y, 0.07, YELLOW if success else RED)
        top = y + 0.07
        # Left: the stats table.
        size = 0.046
        self._block(
            out,
            ids[9:16],
            self._score_rows(sm),
            0.015,
            top,
            size,
            alpha=min(1.0, max(0.0, (e - 0.3) / 0.3)),
        )
        # Right: the final score counts up, then the prompt blinks.
        score = max(0, min(99999, _int(sm.get("score"))))
        shown_score = int(round(score * _ease_out((e - 0.4) / 1.4))) if e > 0.4 else 0
        caption = "FINAL SCORE"
        self._text(
            out, ids[16:18], caption, 0.985 - self._mono_w(len(caption), 0.05), top, 0.05, DIM, MONO
        )
        number = f"{shown_score}"
        nsize = 0.2
        self._text(
            out,
            ids[18:21],
            number,
            0.985 - self.text_width(number, nsize, SANS),
            top + 0.025,
            nsize,
            YELLOW,
            SANS,
            bold=True,
        )
        if e > 2.0:  # (a gentle pulse: the prompt must stay readable in the submission screenshot)
            pulse = 0.82 + 0.18 * (0.5 + 0.5 * math.cos(2 * math.pi * 0.9 * e))
            text = _ascii(sm.get("prompt") or "[R] RETRY   [N] NEW MISSION")
            rows = [self._key_segments(part.strip()) for part in text.split("   ") if part.strip()]
            chars = max(sum(len(seg) for seg, _ in row) for row in rows)
            psize = self._fit("M" * chars, 0.056, MONO, 0.4, 0.04)
            px = 0.985 - self._mono_w(chars, psize)
            self._block(
                out, ids[21:24], rows, px, top + 0.025 + 0.5 * nsize + 0.02, psize, alpha=pulse
            )

    def _compose_experiment_board(self, out: dict[int, Label]) -> None:
        """A test's results: the headline and the table that goes into the reflection."""
        sm = self._summary
        ids = list(_PANEL)
        e = self._t - self._screen_t
        head = _ascii(sm.get("headline") or "EXPERIMENT COMPLETE").upper()
        y = self._panel_title(out, ids[0:3], head, 0.035, 0.13, GREEN, alpha=min(1.0, e / 0.25))
        top = y + 0.03
        rows = self._experiment_rows(sm)
        chars = max(sum(len(seg) for seg, _ in row) for row in rows)
        size = self._fit("M" * chars, 0.05, MONO, 0.97, 0.03)  # (a wide table: a smaller font)
        self._block(
            out, ids[9:16], rows, 0.015, top, size, alpha=min(1.0, max(0.0, (e - 0.3) / 0.3))
        )
        if e > 1.0:
            text = _ascii(sm.get("prompt") or "[R] REPEAT   [N] NEW CODE")
            rows = [self._key_segments(part.strip()) for part in text.split("   ") if part.strip()]
            chars = max(sum(len(seg) for seg, _ in row) for row in rows)
            psize = self._fit("M" * chars, 0.056, MONO, 0.4, 0.04)
            self._block(out, ids[21:24], rows, 0.985 - self._mono_w(chars, psize), top, psize)

    @staticmethod
    def _metric_text(value, unit: str) -> str:
        """A table cell: 0.45 m, 95 m, 76.0 s, 41 %, 27 (a count), COMPLETE (a text), or -
        (none).
        """
        if isinstance(value, str):
            return _ascii(value)[:10]
        v = _num(value)
        if v is None:
            return "-"
        if unit == "m":
            return f"{v:.0f} m" if v >= 10 and float(v).is_integer() else f"{min(v, 99.0):.2f} m"
        if unit == "s":
            return f"{v:.1f} s"
        if unit == "%":
            return f"{v:.0f} %"
        return f"{v:.0f}"

    @staticmethod
    def _experiment_rows(sm: dict) -> list[list[tuple[str, int]]]:
        """Rows of a test table: every test of this session's experiment group (the one just run
        marked >) with its metrics (``experiment["metrics"]``: name, value, unit); the first metric
        also as a bar; "-" for a test not run yet. Sensor tests: spread (the size of the particle
        cloud) and error; particle tests: error and the error of the first 6 s; penalty tests:
        safety
        stops, time, and the share of moves next to shelves or walls.
        """
        ex = sm.get("experiment") if isinstance(sm.get("experiment"), dict) else {}
        robot = _ascii(sm.get("robot_name") or (sm.get("profile") or {}).get("name") or "")[:16]
        tests = [t for t in list(ex.get("session") or []) if isinstance(t, dict)] or [
            dict(ex, current=True)
        ]
        names = [(str(m[0]), str(m[2])) for m in list(ex.get("metrics") or []) if len(m) == 3][:5]

        def value(test: dict, k: int):
            metrics = [m for m in list(test.get("metrics") or []) if len(m) == 3]
            if k >= len(metrics):
                return None
            return metrics[k][1] if isinstance(metrics[k][1], str) else _num(metrics[k][1])

        wide = len(names) > 3  # (a PENALTY TEST: five columns, no bar)
        col = 10 if wide else 12
        width = 19 + (8 if wide else 18) + col * max(0, len(names) - 1)
        rows: list[list[tuple[str, int]]] = []
        if robot:
            rows.append([("ROBOT    ", DIM), (robot, WHITE)])
        rows.append([("-" * width, FAINT)])
        if names:
            firsts = [value(t, 0) for t in tests]
            top = max([v for v in firsts if isinstance(v, (int, float))] + [1e-9])
            first_w = 8 if wide else 18
            header = f"{'   TEST':<19}{_ascii(names[0][0])[: first_w - 1]:<{first_w}}" + "".join(
                f"{_ascii(n)[: col - 1]:>{col}}" for n, _ in names[1:]
            )
            rows.append([(header, DIM)])
            for n, (test, first) in enumerate(zip(tests, firsts, strict=False), start=1):
                color = WHITE if test.get("metrics") else FAINT
                row = [
                    (">" if test.get("current") else " ", GREEN),
                    (f"{n}. {_ascii(test.get('label'))[:14]:<14} ", color),
                    (f"{Hud._metric_text(first, names[0][1]):<7}", CYAN),
                ]
                if not wide:
                    bar = (
                        ""
                        if not isinstance(first, (int, float))
                        else "#" * max(1 if first > 0 else 0, int(round(10 * first / top)))
                    )
                    row.append((f"{bar:<11}", CYAN))
                else:
                    row.append((" ", CYAN))
                for k, (_, unit) in enumerate(names[1:], start=1):
                    row.append((f"{Hud._metric_text(value(test, k), unit):>{col}}", YELLOW))
                if _int(test.get("lost")) > 0:
                    row.append((f"  LOST {_int(test.get('lost'))}", WHITE))
                rows.append(row)
        rows.append([("-" * width, FAINT)])
        for line in {
            "sensors": [
                "SPREAD = size of the particle cloud (sigma)",
                "ERROR  = mean |estimate - truth|",
            ],
            "start": [
                "ERROR = mean |estimate - truth|",
                "FIRST 6 S = the mean error in the first 6 s",
                "SPREAD = size of the particle cloud (sigma)",
            ],
            "penalty": [
                "ROUTE, PROX ON = the planned route (rack + dock legs)",
                "and its cells next to a shelf or wall (proximity ON)",
                "STOPS = safety stops while driving it",
                "WHITE = this route, ORANGE = the other test's route",
            ],
        }.get(str(ex.get("group")), []):
            rows.append([(line, DIM)])
        rows.append(
            [
                (
                    "Every run has the same path and noise."
                    if ex.get("group") != "penalty"
                    else "Every run has the same start, order, and noise.",
                    DIM,
                )
            ]
        )
        return rows

    @staticmethod
    def _key_segments(text: str) -> list[tuple[str, int]]:
        """Color the key names of a prompt ("[R] RETRY", "R: REPLAY") yellow and the rest white."""
        segments: list[tuple[str, int]] = []
        for k, word in enumerate(text.split(" ")):
            if k:
                segments.append((" ", WHITE))
            is_key = (len(word) == 3 and word[0] == "[" and word[2] == "]") or (
                len(word) == 2 and word[1] == ":"
            )
            segments.append((word, YELLOW if is_key else WHITE))
        return segments

    @staticmethod
    def _score_rows(sm: dict) -> list[list[tuple[str, int]]]:
        """Rows of the score table (33 columns): caption, value and the points it earned or cost."""
        terms = sm.get("score_terms") if isinstance(sm.get("score_terms"), dict) else {}
        robot = _ascii(sm.get("robot_name") or (sm.get("profile") or {}).get("name") or "")[:16]
        width = 33
        rows: list[list[tuple[str, int]]] = []
        if robot:
            rows.append([("ROBOT    ", DIM), (robot, WHITE)])
        if _num(sm.get("code")) is not None:
            rows.append([("MISSION  ", DIM), (f"{_int(sm.get('code')) % 10000:04d}", YELLOW)])
        rows.append([("-" * width, FAINT)])

        def row(caption: str, value: str, points=None) -> None:
            segments = [(f"{caption:<17}", DIM), (f"{value:<9}", WHITE)]
            if _num(points) is not None:
                p = max(-99999, min(99999, _int(points)))
                color = GREEN if p > 0 else (RED if p < 0 else FAINT)
                segments.append((f"{p:+7d}" if p else f"{0:7d}", color))
            rows.append(segments)

        if _num(sm.get("time")) is not None:
            row("TIME", f"{min(_num(sm['time']), 9999.0):.1f} s", terms.get("time"))
        if _num(sm.get("distance")) is not None:
            row("DISTANCE", f"{min(_num(sm['distance']), 99999.0):.0f} m")
        if _int(sm.get("orders")) > 0:
            row("DELIVERIES", f"{_int(sm.get('orders_completed'))} / {_int(sm.get('orders'))}")
        row("SAFETY STOPS", str(_int(sm.get("safety_stops"))), terms.get("safety_stops"))
        row("FAILED DOCKINGS", str(_int(sm.get("failed_dockings"))), terms.get("failed_dockings"))
        row("RELOCALIZATIONS", str(_int(sm.get("relocalizations"))), terms.get("relocalizations"))
        error = _num(sm.get("mean_error"))
        row(
            "TRACKING ERROR",
            "-" if error is None else f"{min(error, 99.0):.2f} m",
            terms.get("tracking"),
        )
        smoother = sm.get("smoother") if isinstance(sm.get("smoother"), dict) else {}
        smoothed = _num(smoother.get("smooth_error"))
        if smoothed is not None:  # TODO 17: the post-mission Kalman smoother (not scored)
            row("SMOOTHED PATH", f"{min(smoothed, 99.0):.2f} m")
        for key, caption in (("kidnaps", "KIDNAPS SURVIVED"), ("obstacles", "OBSTACLES")):
            if _int(sm.get(key)) > 0:  # chaos events of the K / O keys
                row(caption, str(_int(sm.get(key))))
        if terms.get("base") is not None:
            row("BASE", "", terms.get("base"))
        rows.append([("-" * width, FAINT)])
        return rows

    def _compose_fault(self, out: dict[int, Label]) -> None:
        """What went wrong, which TODO to fix, and how to retry."""
        f = self._fault
        ids = list(_PANEL)
        t = self._t
        reason = str(f.get("reason") or "")
        # A boot fault lists the missing TODOs; a mid-mission StudentFault stores a bool here
        # instead.
        m = f.get("missing")
        missing = list(m) if isinstance(m, (list, tuple)) else []
        if f.get("title"):
            head = _ascii(f["title"]).upper()
        elif reason == "missing_todos" or missing:
            head = "MISSING TODOs"
        elif reason == "exception":
            head = "STUDENT CODE ERROR"
        elif "import" in reason or "load" in reason or "syntax" in reason:
            head = "CANNOT LOAD project4.py"
        else:
            head = "SYSTEM FAULT"
        pulse = 0.7 + 0.3 * (0.5 + 0.5 * math.cos(2 * math.pi * 1.0 * t))
        y = self._panel_title(out, ids[0:3], f"!! {head} !!", 0.08, 0.12, RED, alpha=pulse)
        cols = 58
        rows: list[list[tuple[str, int]]] = []
        if missing:
            rows.append([("The robot cannot deploy: implement these TODOs", WHITE)])
            rows.append([("in project4.py (python -m pytest -q helps):", WHITE)])
            rows.append([("", WHITE)])
            for item in missing[:12]:
                todo, name = (
                    (tuple(item) + ("", ""))[:2] if isinstance(item, (tuple, list)) else (item, "")
                )
                rows.append(
                    [
                        (f"  TODO {_int(todo):>2}  ", DIM),
                        (f"{_ascii(name)[:24]:<24}", WHITE),
                        ("[MISSING]", RED),
                    ]
                )
        else:
            function, todo = f.get("function"), f.get("todo")
            if function:
                where = _ascii(f"in {function}" + (f" (TODO {todo})" if todo else ""))[:cols]
                rows.append([(where, YELLOW)])
            for line in _wrap(f.get("message") or "unknown error", cols, 5):
                rows.append([(line, WHITE)])
            rows.append([("", WHITE)])
            rows.append([("The full traceback is in the Webots console.", DIM)])
        rows.append([("", WHITE)])
        hint = f.get("hint") or "Fix project4.py, save it, then reload the world (Ctrl+Shift+R)."
        for k, line in enumerate(_wrap(hint, cols - 2, 3)):
            rows.append([("> " if k == 0 else "  ", CYAN), (line, CYAN)])
        size = 0.05
        width = self._mono_w(cols, size)
        x = max(0.02, 0.5 - width / 2)
        self._block(out, ids[3:9], rows, x, y + 0.03, size)
        prompt = "[R] RETRY"
        py = min(0.69, y + 0.03 + 0.5 * size * len(rows) + 0.03)
        self._text(
            out,
            ids[9:11],
            prompt,
            0.5 - self._mono_w(len(prompt), 0.06) / 2,
            py,
            0.06,
            YELLOW,
            MONO,
        )

    def _compose_help(self, out: dict[int, Label]) -> None:
        """Two columns of keys (yellow) and what they do; reloading hint at the bottom."""
        ids = list(_HELP)
        half = (len(_HELP_KEYS) + 1) // 2
        rows: list[list[tuple[str, int]]] = []
        for k in range(half):
            row: list[tuple[str, int]] = []
            for key, what in (
                _HELP_KEYS[k],
                _HELP_KEYS[k + half] if k + half < len(_HELP_KEYS) else ("", ""),
            ):
                if key:
                    row += [(f"{key:>5}  ", YELLOW), (f"{what:<18}", WHITE)]
                row.append(("   ", WHITE))
            rows.append(row)
        rows.append([("", WHITE)])
        rows.append([("Click the 3D view first. Edited project4.py? Use", DIM)])
        rows.append([("File > Reload World (Ctrl/Cmd+Shift+R).", DIM)])
        size = 0.052
        cols = max(len("".join(seg for seg, _ in row)) for row in rows)
        width = self._mono_w(cols, size)
        x = max(0.35, 0.5 - width / 2) if 0.35 + width < 0.99 else max(0.01, 0.99 - width)
        y = self._panel_title(out, ids[0:3], "CONTROLS", 0.2, 0.09, CYAN, left=x)
        self._block(out, ids[3:7], rows, x, y + 0.02, size)
