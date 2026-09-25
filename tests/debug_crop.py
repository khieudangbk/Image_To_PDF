"""Shows why document-edge detection accepts/rejects each candidate contour for one image."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.preprocess import find_document_quad, load_image  # noqa: E402


def main(path: str):
    bgr = load_image(path)
    print("image", bgr.shape)
    quad = find_document_quad(bgr)
    print("quad:", None if quad is None else quad.round().tolist())
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    vis = bgr.copy()
    if quad is not None:
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 0, 255), 6)
    cv2.imwrite(str(out / (Path(path).stem + "_crop.jpg")), vis)


if __name__ == "__main__":
    main(sys.argv[1])
