import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "vision_samples"
SIDE = 896
DPI = 112
FIG = SIDE / DPI


def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=DPI, format="jpeg", pil_kwargs={"quality": 85})
    plt.close(fig)


def bar_chart():
    fig, ax = plt.subplots(figsize=(FIG, FIG))
    labels = ["Q1", "Q2", "Q3", "Q4"]
    ax.bar(labels, [42, 58, 51, 74], color=["#2f6fdb", "#3b9c6e", "#e0a030", "#d2553f"])
    ax.set_title("Quarterly revenue (millions)")
    ax.set_ylabel("Revenue")
    save_fig(fig, "bar_chart.jpg")


def line_chart():
    fig, ax = plt.subplots(figsize=(FIG, FIG))
    x = list(range(0, 24))
    ax.plot(x, [20 + 12 * math.sin(i / 3.5) + i * 0.8 for i in x], label="Requests", color="#2f6fdb", linewidth=2.5)
    ax.plot(x, [15 + 8 * math.cos(i / 4) + i * 0.4 for i in x], label="Errors", color="#d2553f", linewidth=2.5)
    ax.set_title("Traffic over one day")
    ax.set_xlabel("Hour")
    ax.legend()
    save_fig(fig, "line_chart.jpg")


def pie_chart():
    fig, ax = plt.subplots(figsize=(FIG, FIG))
    ax.pie([38, 27, 20, 15], labels=["Search", "Social", "Direct", "Email"], autopct="%1.0f%%",
           colors=["#2f6fdb", "#3b9c6e", "#e0a030", "#8a63d2"])
    ax.set_title("Traffic sources")
    save_fig(fig, "pie_chart.jpg")


def scatter_plot():
    rng = random.Random(7)
    fig, ax = plt.subplots(figsize=(FIG, FIG))
    for color, cx, cy in (("#2f6fdb", 2, 2), ("#d2553f", 6, 5), ("#3b9c6e", 3, 7)):
        ax.scatter([rng.gauss(cx, 0.8) for _ in range(60)], [rng.gauss(cy, 0.8) for _ in range(60)], color=color, s=28, alpha=0.8)
    ax.set_title("Three clusters")
    save_fig(fig, "scatter_clusters.jpg")


def heatmap():
    rng = random.Random(3)
    fig, ax = plt.subplots(figsize=(FIG, FIG))
    data = [[rng.random() * (1 + r * 0.15) for _ in range(10)] for r in range(10)]
    ax.imshow(data, cmap="viridis")
    ax.set_title("Utilization heatmap")
    save_fig(fig, "heatmap.jpg")


def landscape():
    img = Image.new("RGB", (SIDE, SIDE))
    d = ImageDraw.Draw(img)
    for y in range(SIDE):
        t = y / SIDE
        d.line([(0, y), (SIDE, y)], fill=(int(250 - 90 * t), int(170 - 40 * t), int(90 + 110 * t)))
    d.ellipse([620, 130, 760, 270], fill=(255, 224, 120))
    d.polygon([(0, 640), (200, 380), (380, 600), (520, 430), (760, 650), (896, 520), (896, 896), (0, 896)], fill=(60, 78, 110))
    d.polygon([(0, 760), (260, 600), (500, 740), (720, 620), (896, 740), (896, 896), (0, 896)], fill=(32, 52, 44))
    img.save(OUT / "sunset_mountains.jpg", quality=85)


def flowchart():
    img = Image.new("RGB", (SIDE, SIDE), (248, 248, 244))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    boxes = [("Client", 110), ("Admission", 300), ("Queue", 490), ("GPU", 680)]
    for i, (label, y) in enumerate(boxes):
        d.rounded_rectangle([280, y, 616, y + 110], radius=18, fill=(220, 232, 250), outline=(47, 111, 219), width=4)
        d.text((448, y + 55), label, fill=(20, 40, 90), font=font, anchor="mm")
        if i < len(boxes) - 1:
            d.line([(448, y + 110), (448, y + 190)], fill=(47, 111, 219), width=5)
            d.polygon([(448, y + 196), (432, y + 170), (464, y + 170)], fill=(47, 111, 219))
    img.save(OUT / "request_flow.jpg", quality=85)


def geometric():
    img = Image.new("RGB", (SIDE, SIDE), (18, 22, 40))
    d = ImageDraw.Draw(img)
    for i in range(14):
        r = 430 - i * 30
        color = (40 + i * 14, 90 + i * 8, 220 - i * 10)
        d.ellipse([448 - r, 448 - r, 448 + r, 448 + r], outline=color, width=6)
    for i in range(12):
        a = i * math.pi / 6
        d.line([(448, 448), (448 + 420 * math.cos(a), 448 + 420 * math.sin(a))], fill=(230, 180, 60), width=3)
    img.save(OUT / "concentric_rings.jpg", quality=85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (bar_chart, line_chart, pie_chart, scatter_plot, heatmap, landscape, flowchart, geometric):
        build()


if __name__ == "__main__":
    main()
