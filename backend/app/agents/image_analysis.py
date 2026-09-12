"""
Image Analysis Agent.

Owner: teammate.

Detects accessibility features (ramps, stairs, handrails, signage, etc.)
from photos of a location. Combines a fine-tuned YOLOv8n detector (for the
5 classes we have training data for) with Gemini's vision model as a
zero-shot fallback for the rest — see the project's implementation plan
doc for the full reasoning.

RULES FOR THIS FILE (so integration back into main stays painless):
  1. No database calls in here. Take ImageAnalysisInput, return
     ImageAnalysisOutput. Whoever calls this function handles persistence.
  2. Never invent a "not detected" when there's genuinely no evidence —
     use `detected=None` (meaning "not verified"). This matches the
     project's core design principle: missing evidence != inaccessible.
  3. Keep external calls (downloading images, calling Gemini) wrapped in
     try/except so one bad image URL doesn't crash the whole batch.
"""

from app.agents.contracts import (
    ImageAnalysisInput,
    ImageAnalysisOutput,
    FeatureFinding,
    ACCESSIBILITY_FEATURE_CLASSES,
)


def analyze_images(input: ImageAnalysisInput) -> ImageAnalysisOutput:
    """
    PHASE 1 (do this first, day 1-2 of your side):
    Return a stub — every feature "not verified", 0 images processed.
    This lets you commit a working, importable function immediately, and
    lets the rest of the pipeline be wired against the real function
    signature before any CV code exists.

    PHASE 2 (the real work):
    Replace the body below with:
      - Download each image (requests / httpx)
      - Preprocess with OpenCV (resize, blur/quality filter, dedupe)
      - Run the fine-tuned YOLOv8n model on the 5 core classes
      - For classes YOLO doesn't cover, call Gemini's vision API with a
        prompt asking it to identify the feature in the image
      - Merge both sources into `features`, filling in confidence and
        which image(s) supported the finding
    """
    features = {
        cls: FeatureFinding(detected=None, confidence=None, source_images=[])
        for cls in ACCESSIBILITY_FEATURE_CLASSES
    }

    return ImageAnalysisOutput(
        location_id=input.location_id,
        features=features,
        processed_image_count=0,
    )


if __name__ == "__main__":
    # Quick manual test — run this file directly: `python -m app.agents.image_analysis`
    # (from inside backend/, with your venv active)
    test_input = ImageAnalysisInput(
        location_id="loc_test",
        location_name="Test Mall",
        image_urls=["https://example.com/sample1.jpg"],
    )
    result = analyze_images(test_input)
    print(result.model_dump_json(indent=2))
