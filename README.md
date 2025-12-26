# LamaWpf

WPF test UI for LaMa (ONNX) inpainting.

## Features
- Load ONNX model (`*.onnx`)
- Load input image
- Create mask with:
  - **Rect**: drag to add multiple rectangles (accumulates)
  - **Brush**: paint with mouse (accumulates), brush size preview circle on hover
- Run inpainting and show output
- **Clear Mask** clears only the mask (keeps output)

## Build / Run
- Visual Studio 2022
- .NET 8 (x64)

Open `LamaWpf.sln` and run `LamaWpf` project.

## Notes
- Mask is stored as **Gray8** (0/255). Multiple regions are supported because the mask is a single bitmap buffer.
- If you want "erase brush" later, you can add a mode to write `0` instead of `255` in `PaintBrushAt()`.

## Third-party / License Notice
This project uses third-party open-source software and pretrained models.

### 1. LaMa (Original Research & Implementation)
- Repository: https://github.com/advimman/lama
- Paper: *LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions* (WACV 2022)
- License: Apache License 2.0

Copyright information is provided in the original repository’s LICENSE file.

### 2. LaMa-ONNX (ONNX Model Conversion)
- Model page: https://huggingface.co/Carve/LaMa-ONNX
- Model file used:
  - https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx
- License: Apache License 2.0

The ONNX model is redistributed under the terms specified by the model publisher.
Copyright ownership remains with the original authors.

### Disclaimer
This project is not affiliated with, endorsed by, or sponsored by the original authors
or the model publisher.

For full license terms, see:
https://www.apache.org/licenses/LICENSE-2.0
