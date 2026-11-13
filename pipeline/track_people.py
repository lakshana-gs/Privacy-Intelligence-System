import sys
from ultralytics import YOLO

source = sys.argv[1] if len(sys.argv) > 1 else "sample_data/WhatsApp Video 2026-11-12 at 12.30.21.mp4"

model = YOLO("yolo26n.pt")

results = model.track(
    source=source,
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
    show=False,
    classes=[0],
    conf=0.5
)

print("Tracking Complete")