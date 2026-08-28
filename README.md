# FaceGlow

FaceGlow is a lightweight Python project for **offline photo beautification and batch processing**. It does not access a camera and does not process real-time video. InsightFace handles face detection and facial landmarks, while OpenCV and NumPy perform the actual image processing: precise skin-region detection, facial-feature protection, edge-preserving skin smoothing, texture restoration, localized skin brightening, and local cheek/jaw slimming.

FaceGlow can process a single image, an entire folder, or nested subfolders. It can also be imported as a Python module. Source images are treated as read-only: processed results are written to new files or directories, and the program does not overwrite, move, or delete the originals.

## Features

- InsightFace `buffalo_l` face detection.
- InsightFace 106-point facial landmark detection.
- InsightFace 5-point landmarks for facial-feature protection.
- Multi-color-space skin detection using YCrCb, HSV, and Lab.
- Combined 106-point face geometry and skin-color constraints.
- Protection for eyebrows, eyes, lips, and central nose details.
- Edge-preserving bilateral skin smoothing.
- High-frequency texture restoration to reduce an over-smoothed look.
- Local Lab luminance enhancement and mild color neutralization.
- Local gamma-based skin brightening.
- Facial-feature detail restoration.
- Independent local slimming for the left cheek, right cheek, left jaw, and right jaw.
- Single-image processing.
- Folder batch processing.
- Recursive subfolder processing.
- Preservation of relative directory structure in batch output.
- Multi-face processing; all detected faces are processed by default.
- Automatic output-name conflict handling without overwriting existing files.
- Windows Unicode path support.
- Core algorithms are separated into reusable `def` functions for testing, maintenance, and replacement.

## Project Structure

```text
FaceGlow/
|-- beauty_processor.py
|-- README.md
`-- LICENSE
```

## Recommended Environment

Python 3.10, 3.11, or 3.12 is recommended.

Main dependencies:

```text
insightface
onnxruntime
opencv-python
numpy
```

The default configuration uses `CPUExecutionProvider`, so NVIDIA CUDA is not required.

## Installation

Clone the repository:

```bash
git clone https://github.com/wangyifan349/FaceGlow
cd FaceGlow
```

Install the dependencies:

```bash
pip install insightface onnxruntime opencv-python numpy
```

On the first run, InsightFace may download the `buffalo_l` model files.

## Quick Start

### Process One Image

```bash
python beauty_processor.py --input photo.jpg --output beauty_output
```

For an input file such as:

```text
photo.jpg
```

FaceGlow creates an output similar to:

```text
beauty_output/photo_beauty.jpg
```

The original `photo.jpg` is not modified.

### Specify an Output File

```bash
python beauty_processor.py --input photo.jpg --output result.jpg
```

If `result.jpg` already exists, FaceGlow does not overwrite it. A new name is selected automatically, for example:

```text
result_1.jpg
result_2.jpg
```

### Batch Process a Folder

```bash
python beauty_processor.py --input photos --output beauty_output
```

Subfolders are scanned recursively by default.

Example input directory:

```text
photos/
|-- portrait.jpg
|-- group.png
`-- trip/
    `-- image.jpg
```

The relative directory structure is preserved in the output:

```text
beauty_output/
|-- portrait_beauty.jpg
|-- group_beauty.png
`-- trip/
    `-- image_beauty.jpg
```

### Disable Recursive Scanning

```bash
python beauty_processor.py --input photos --output beauty_output --no-recursive
```

## Beautification Strength

```bash
python beauty_processor.py \
  --input photos \
  --output beauty_output \
  --smooth 0.62 \
  --whiten 0.24 \
  --slim 0.16 \
  --detail 0.30
```

Main options:

| Option | Default | Description |
| --- | ---: | --- |
| `--smooth` | `0.62` | Skin smoothing strength. Recommended range: `0.0 - 1.0`. |
| `--whiten` | `0.24` | Local skin brightening strength. Recommended range: `0.0 - 1.0`. |
| `--slim` | `0.16` | Face slimming strength. Recommended range: `0.0 - 1.0`. |
| `--detail` | `0.30` | Skin texture and facial-feature detail restoration strength. |
| `--max-faces` | `0` | Maximum number of faces to process. `0` means all detected faces. |
| `--det-size` | `640` | InsightFace detection input size. |
| `--det-threshold` | `0.50` | Face detection threshold. |
| `--output-suffix` | `_beauty` | Suffix added before the output file extension. |
| `--no-recursive` | disabled | Do not scan subfolders. |

For example, process only the largest detected face:

```bash
python beauty_processor.py \
  --input photo.jpg \
  --output beauty_output \
  --max-faces 1
```

## Image Processing Pipeline

FaceGlow does not apply a global filter to the entire image. It first locates the face and estimates the skin regions that should actually be processed.

### 1. Face Detection

InsightFace detects faces and provides a bounding box for each detected face.

### 2. 106-Point Facial Landmarks

The `landmark_2d_106` model provides facial landmarks used for face geometry, masks, and slimming anchors.

### 3. Geometric Face Region

`create_geometry_face_mask()` intersects the 106-point convex hull with a conservative face ellipse. This helps prevent hair, background, and areas outside the face from entering the beautification mask.

### 4. Multi-Color-Space Skin Detection

`create_skin_color_mask()` evaluates skin candidates in three color spaces:

- YCrCb
- HSV
- Lab

The results are combined by voting instead of relying on a single fixed color-space threshold.

### 5. Facial-Feature Protection

`create_feature_protection_mask()` uses the 5-point landmarks to protect important details around:

- Left and right eyes
- Eyebrow regions
- Lips
- Central nose details

This reduces the chance of smoothing away sharp facial features.

### 6. Precise Skin Mask

`create_precise_skin_mask()` combines the geometric face region, skin-color region, and facial-feature protection mask, then feathers the mask edges.

As a result, smoothing and brightening mainly affect facial skin instead of the entire photograph.

### 7. Skin Smoothing

`smooth_skin()` uses bilateral filtering for edge-preserving smoothing. High-frequency texture is then extracted from the original image and partially restored.

This reduces small skin irregularities and noise while retaining more natural texture.

### 8. Skin Brightening

`whiten_skin()` includes:

- Lab luminance enhancement.
- Highlight-aware brightness adjustment.
- Mild Lab chroma neutralization.
- Soft gamma correction.
- Blending through the precise skin mask.

Background regions, hair, and most facial features are therefore not brightened together with the skin.

### 9. Facial-Feature Detail Restoration

`restore_feature_details()` applies lightweight unsharp restoration to protected facial features and contour areas after smoothing.

### 10. Face Slimming

`slim_face()` does not resize the entire face. It finds separate local anchors for:

- Left cheek
- Right cheek
- Left jaw
- Right jaw

`local_translation_warp()` then performs a local inverse warp with radial falloff, gently pulling the face contour inward.

Geometric warping is applied after smoothing, brightening, and detail restoration so that the original landmark positions and masks remain aligned during those operations.

## Use as a Python Module

For repeated processing, create one `BeautyProcessor` instance and reuse it. This allows the InsightFace models to be initialized only once for a batch job.

```python
from beauty_processor import BeautyConfig, BeautyProcessor

config = BeautyConfig(
    smooth_strength=0.62,
    whiten_strength=0.24,
    slim_strength=0.16,
    detail_strength=0.30,
    max_faces=0,
)

processor = BeautyProcessor(config)
```

### Process a NumPy / OpenCV Image

```python
import cv2

from beauty_processor import BeautyProcessor

processor = BeautyProcessor()
source_image = cv2.imread("photo.jpg")
beautified_image, processed_face_count = processor.beautify_image(source_image)

print(processed_face_count)
```

`beautify_image()` does not modify the input NumPy array in place. It returns a new processed image.

### Process One File

```python
from beauty_processor import BeautyProcessor, process_image_file

processor = BeautyProcessor()
processing_result = process_image_file(
    input_path="photo.jpg",
    output="beauty_output",
    processor=processor,
)

print(processing_result.output_path)
print(processing_result.face_count)
```

### Batch Process a Folder

```python
from beauty_processor import BeautyProcessor, process_batch

processor = BeautyProcessor()
batch_summary = process_batch(
    input_directory="photos",
    output_directory="beauty_output",
    processor=processor,
    recursive=True,
)

print(batch_summary.total_files)
print(batch_summary.success_files)
```

### Unified File-or-Directory Entry Point

```python
from beauty_processor import BeautyProcessor, process_path

processor = BeautyProcessor()
processing_result = process_path(
    input_path="photos",
    output="beauty_output",
    processor=processor,
)
```

## Reusable Core Functions

```text
create_geometry_face_mask()       Build the 106-point geometric face mask
create_skin_color_mask()          Detect skin using multiple color spaces
create_feature_protection_mask()  Build the facial-feature protection mask
create_precise_skin_mask()        Build the final precise skin mask
blend_with_mask()                 Blend images with a soft mask
smooth_skin()                     Smooth skin and restore high-frequency texture
whiten_skin()                     Apply localized skin brightening
restore_feature_details()         Restore facial-feature details
choose_side_anchor()              Select cheek/jaw deformation anchors
local_translation_warp()          Apply local radial image deformation
slim_face()                       Slim cheeks and jaw regions
beautify_image()                  Functional NumPy image entry point
process_image_file()              Single-file processing entry point
process_batch()                   Folder batch-processing entry point
process_path()                    Unified file/directory entry point
```

These functions are intentionally layered. Replacing the skin-smoothing, skin-detection, brightening, or face-slimming algorithm does not require rewriting batch scanning or file I/O code.

## Original-File Safety

FaceGlow follows a non-destructive output policy.

- Input images are read only.
- Input files are not deleted.
- Input files are not moved.
- Processed results are never written directly over the input file.
- Output-name conflicts are resolved by selecting a new filename automatically.
- A batch output directory cannot be exactly the same directory as the input directory.
- If the output directory is inside the input directory, it is excluded from the current scan to prevent generated files from being processed again.
- Batch jobs finish building the input file list before writing results.

Recommended usage:

```bash
python beauty_processor.py --input photos --output beauty_output
```

Using the source-photo directory itself as the output location is not recommended, and FaceGlow rejects an identical input/output directory for batch jobs.

## Supported Output Formats

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

Default JPEG quality is `95`. Default PNG compression level is `3`.

Output images are re-encoded. FaceGlow currently does not guarantee preservation of all EXIF, GPS, ICC profile, or camera-vendor private metadata in the generated files. This does not affect the original photographs because the source files are never modified.

## Model and License Notes

Source code written for this repository is released under the MIT License. See [LICENSE](LICENSE).

The MIT License covers only the source code in this repository. It does not change the licenses of InsightFace, ONNX Runtime, OpenCV, NumPy, or third-party pretrained models. InsightFace pretrained models must be used according to their own official model license and terms.

This project is primarily intended for non-commercial learning, research, and open-source experimentation. Before using it in a commercial product, verify that the selected InsightFace models and all other third-party components provide the required commercial permissions.

## License

MIT License.

Copyright (c) 2026 wangyifan349
