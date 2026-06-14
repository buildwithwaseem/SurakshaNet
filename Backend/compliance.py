"""
SurakshaNet - Compliance Classification Module
Single source of truth for turning raw YOLO detections into
SAFE / BREACH / EQUIPMENT statuses, using IoU overlap between
Head/Person boxes and Helmet boxes.
"""


def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def classify_compliance(detections, iou_thresh: float = 0.3):
    """
    detections: list of {"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
    Returns the same items annotated with a "status" key:
        - "SAFE"      -> Head/Person covered by a Helmet box
        - "BREACH"    -> Head/Person with no overlapping Helmet
        - "EQUIPMENT" -> Helmet box itself (informational, not a violation)
    """
    heads = [d for d in detections if d["class"] in ("Head", "Person")]
    helmets = [d for d in detections if d["class"] == "Helmet"]

    results = []
    for h in heads:
        covered = any(iou(h["bbox"], hel["bbox"]) > iou_thresh for hel in helmets)
        results.append({**h, "status": "SAFE" if covered else "BREACH"})
    for hel in helmets:
        results.append({**hel, "status": "EQUIPMENT"})
    return results
