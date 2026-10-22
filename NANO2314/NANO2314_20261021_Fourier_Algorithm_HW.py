# NANO 2314 homework, October 21, 2026 lecture, Problem 3:
# the Fourier transform, one frequency at a time.
#
# HOW TO USE: open Google Colab (https://colab.research.google.com), start a new notebook,
# paste this whole file into a code cell, and run it (Shift + Enter).
#
# What it does, for ONE reference frequency f (the same steps as Problem 2, on real data):
#   1. signal      x(t): cantilever deflection, 500 µs of data
#   2. references  sin(2π f t) and cos(2π f t)
#   3. products    x(t) · sin(2π f t) and x(t) · cos(2π f t)
#   4. areas       A_sin and A_cos under the two product curves, over the whole record
#                  (T_window = 500 µs)
#   5. amplitude   = 2 √(A_sin² + A_cos²) / T_window, in nm.
#      Why both? The sine reference alone finds a vibration only if it is in step with the
#      sine. The cosine catches the part that is a quarter cycle out of step (lecture §4:
#      e^(−i2πft) = cos(2πft) − i sin(2πft)), so together they find it whatever its phase.
#
#   The two combine into one "matched" reference, sin(2πft + φ): the sine shifted in time to
#   line up best with the signal (cos φ = A_sin/√(A_sin² + A_cos²), sin φ = A_cos/√(A_sin² + A_cos²)).
#   The area under x(t) · sin(2πft + φ) is exactly √(A_sin² + A_cos²), so
#   amplitude = 2 × (matched area) / T_window: the same "2 × area / T" as with one sine.
#   A vibration of amplitude A at exactly f gives a matched area of A·T_window/2, so the
#   formula gives back A.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlopen

# ---- the data: cantilever deflection (nm) every 0.1 µs for 500 µs ----
# If you uploaded the file to Colab (folder icon on the left), it is used;
# otherwise it is downloaded from the class web page.
CSV_NAME = "261026-bimodal_AFM-data-v2.csv"
DATA_URL = "https://semori-san.github.io/class-notes/NANO2314/261026-bimodal_AFM-data-v2.csv"
if Path(CSV_NAME).exists():
    source = CSV_NAME
else:
    try:
        source = urlopen(DATA_URL)
    except OSError:                   # no internet, or the class web page can't be reached
        raise SystemExit(f"Couldn't download the data. Upload {CSV_NAME} to Colab "
                         "(folder icon on the left, then the upload button) and run this cell again.") from None
t, x = np.loadtxt(source, delimiter=",", skiprows=1, unpack=True)   # t in µs, x(t) in nm

dt = t[1] - t[0]                      # time step, 0.1 µs
T_window = len(t) * dt                # record length (the measurement window), 500 µs
print(f"{len(t)} points, one every {dt:.1f} µs, T_window = {T_window:.0f} µs")


# ---- one reference frequency ----
def test_frequency(f_kHz, show=False, show_us=None):
    """Multiply the signal by a sine and a cosine reference at f_kHz, add up the two areas,
    and return the amplitude in nm.

    It plots nothing unless you ask: with show=True it also plots the signal, the two
    references, and the sine, cosine and matched products, over the whole record or over
    the first show_us microseconds if you give it (e.g. show_us=50 to see the individual
    cycles). Keep show=False in loops: each plot is about 200 kB, and thousands of them
    would freeze the notebook.
    """
    phase = 2 * np.pi * f_kHz * 1e-3 * t                # kHz × µs = 10^-3, so scale by 1e-3
    A_sin = np.sum(x * np.sin(phase)) * dt               # area under x(t)·sin(2πft), nm·µs
    A_cos = np.sum(x * np.cos(phase)) * dt               # area under x(t)·cos(2πft), nm·µs
    A_matched = np.sqrt(A_sin**2 + A_cos**2)             # area under x(t)·sin(2πft + φ), nm·µs
    amplitude = 2 * A_matched / T_window                 # nm

    if show:
        phi = np.arctan2(A_cos, A_sin)                   # the shift that lines the sine up best
        w = t <= (show_us if show_us else T_window)
        rows = [("sine\nproduct (nm)", x * np.sin(phase), f"$A_{{\\sin}}$ = {A_sin:.1f} nm·µs"),
                ("cosine\nproduct (nm)", x * np.cos(phase), f"$A_{{\\cos}}$ = {A_cos:.1f} nm·µs"),
                ("matched\nproduct (nm)", x * np.sin(phase + phi),
                 f"area = $\\sqrt{{A_{{\\sin}}^2 + A_{{\\cos}}^2}}$ = {A_matched:.1f} nm·µs")]
        fig, ax = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
        ax[0].plot(t[w], x[w], color="tab:blue", lw=0.8)
        ax[0].set_ylabel("signal x(t)\n(nm)")
        ax[1].plot(t[w], np.sin(phase)[w], color="tab:orange", lw=0.8, label="sine")
        ax[1].plot(t[w], np.cos(phase)[w], color="tab:purple", lw=0.8, ls="--", label="cosine")
        ax[1].set_ylabel(f"references\n{f_kHz:g} kHz")
        ax[1].set_ylim(-1.2, 2.4)                        # room for the legend
        ax[1].legend(loc="upper right", ncol=2, fontsize="small")
        x_max = np.abs(x).max()                          # |product| <= |x|, since |reference| <= 1
        for a, (label, product, note) in zip(ax[2:], rows):
            a.plot(t[w], product[w], color="black", lw=0.3)
            a.fill_between(t[w], product[w], where=product[w] > 0, interpolate=True, color="tab:green", alpha=0.5, label="+ area")
            a.fill_between(t[w], product[w], where=product[w] < 0, interpolate=True, color="tab:red", alpha=0.5, label="− area")
            a.set_ylabel(label)
            a.set_ylim(-1.1 * x_max, 2.0 * x_max)        # same scale in every panel and at every f
            a.text(0.01, 0.95, note, transform=a.transAxes, va="top")
        ax[2].legend(loc="upper right", ncol=2, fontsize="small")
        ax[-1].set_xlabel(f"time (µs), first {show_us:g} of {T_window:.0f} µs shown" if show_us else "time (µs)")
        for a in ax:
            a.axhline(0, color="gray", lw=0.6)
        fig.suptitle(f"f = {f_kHz:g} kHz", fontweight="bold")
        ax[0].set_title(f"amplitude = 2·(matched area)/$T_{{\\mathrm{{window}}}}$ = {amplitude:.2f} nm",
                        fontsize="medium")
        fig.tight_layout()
        plt.show()
    return amplitude


# ---- examples: a 68 kHz reference (close, but no match) and a 72 kHz reference (a match) ----
for f_example in (68, 72):
    amplitude = test_frequency(f_example, show=True)
    print(f"{f_example:g} kHz: amplitude = {amplitude:.2f} nm")
