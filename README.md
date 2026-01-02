# WpfAiRunner

A high-performance WPF application for running local AI models via **ONNX Runtime**.
This project demonstrates a production-ready implementation of **LaMa (Inpainting)**, **Depth Anything V2 (Depth Estimation)**, **Segment Anything (MobileSAM & SAM 2)**, **Real-ESRGAN (Super Resolution)**, and **RMBG-1.4 (Background Removal)** with hybrid CPU/GPU execution support.

## ✨ Key Features

### Architecture & Performance
- **Modular Multi-Model UI**: Features a sidebar menu to switch between different model views dynamically.
- **Unified AI Core**: All model inference logic is consolidated into a single `OnnxEngines` library, sharing optimized tensor processing and GPU session management utilities.
- **Hybrid Execution**: Supports both **CPU** and **GPU (CUDA)** with a run-time toggle switch.
- **Smart Fallback**: Automatically falls back to CPU if GPU initialization fails.
- **Optimization**: Includes **GPU Warm-up** logic to eliminate initial inference latency and async processing to prevent UI freezing.

### 1. LaMa (Inpainting)
- **Smart Preprocessing**: Automatically crops and resizes the ROI to `512x512`, then pastes the result back to the original resolution.
- **Masking Tools**:
  - **Rect**: Drag to create rectangular masks.
  - **Brush**: Freehand masking with adjustable brush size.

### 2. Depth Anything V2 (Depth Estimation)
- **Visualisation**: Converts model output (disparity) into a normalized grayscale depth map (Brighter = Near, Darker = Far).
- **Fast Inference**: Supports the **V2 Small** model for real-time performance.

### 3. Segment Anything (SAM & SAM 2)
- **Unified Interface**: Supports both **MobileSAM** and **SAM 2** models within a single view.
- **Workflow**:
  1. **Image Encoding**: When an image is opened, the **Encoder** runs immediately to generate embeddings (resized to 1024px).
  2. **Point Prompting**: Clicking any object in the viewer sends the coordinate prompts to the engine.
  3. **Real-time Decoding**: The **Decoder** uses the pre-calculated embeddings and coordinates to generate a segmentation mask.
  4. **Mask Overlay**: The generated mask is dynamically cropped and resized to match the original image resolution.
- **Automated Encoder-Decoder Matching**: Logic to detect and match Encoder/Decoder pairs based on filenames.
- **Mask Post-processing**: Applies **Bicubic Interpolation** and **Soft Masking** (Sigmoid) when upscaling the raw model output.

### 4. Real-ESRGAN (Super Resolution)
- **x4 Upscaling**: Restores and upscales low-resolution images to 4x their original size.
- **Tiling Strategy**: Implements a sliding window approach with configurable tile size (e.g., 128x128) to process high-resolution images without exceeding memory limits.
- **Seamless Reconstruction**: Uses an **Overlap & Crop** technique (default 14px padding) to eliminate grid artifacts at tile boundaries.
- **Output Management**: Provides progress monitoring during processing and supports saving the result as PNG.

### 5. RMBG-1.4 (Background Removal)
- **State-of-the-Art Segmentation**: Utilizes BRIA AI's RMBG-1.4 model for high-quality background removal, effective even on complex edges like hair and fur.
- **Fine-Tuning Control**: Provides a **Threshold Slider** to adjust the confidence level for masking, allowing users to balance between detail preservation and background noise removal.
- **Background Replacement**: Supports transparent background output as well as instant replacement with solid colors (White, Black, Green Chroma, Blue).
- **Visualization**: Features a checkerboard pattern background in the UI to clearly verify transparency.

## 🛠️ Build & Run

### 1. Prerequisites
- **Visual Studio 2022**
- **.NET 8 SDK**
- **Platform**: Windows x64

### 2. GPU Requirements (Optional)
To enable CUDA acceleration:
- NVIDIA GPU
- **CUDA Toolkit 11.8**
- **cuDNN 8.x** (compatible with CUDA 11.x)
- *Note: If requirements are not met, the app will safely run in CPU mode.*

### 3. Model Setup (Download)
This project requires several large ONNX models. A PowerShell script is provided to download them automatically.

1.  Right-click `download_models.ps1` in the project root.
2.  Select **Run with PowerShell**.
3.  The script will create a `models/` directory and download all required models (LaMa, Depth, SAM, Real-ESRGAN, RMBG).
    * *Note: The script automatically skips files that have already been downloaded.*

### 4. Setup (Build)
1.  Open `WpfAiRunner.sln` in Visual Studio.
2.  Restore NuGet packages.
    - **Important**: This project depends on `Microsoft.ML.OnnxRuntime.Gpu` version **1.15.1**. Do not update to newer versions to ensure compatibility.
3.  Set the build platform to **x64**.
4.  Build and Run the `WpfAiRunner` project.

## 📂 Project Structure

- **WpfAiRunner** (UI): Handles main window, view switching (`Views/`), and user interaction.
- **OnnxEngines** (Core Library): A unified class library containing logic for all AI models.
  - **Utils/**: Shared utilities for Tensor conversion (`TensorHelper`) and ONNX session management (`OnnxHelper`).
  - **Lama/**: Logic for LaMa Inpainting.
  - **Depth/**: Logic for Depth Anything V2.
  - **Sam/**: Logic for MobileSAM & SAM 2.
  - **Upscaling/**: Logic for Real-ESRGAN.
  - **Rmbg/**: Logic for RMBG-1.4 Background Removal.

## ⚖️ License & Acknowledgements

This project uses third-party open-source software and pretrained models.

### LaMa (Large Mask Inpainting)
- **Original Paper**: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
- **Model Source**: [LaMa-ONNX via HuggingFace](https://huggingface.co/Carve/LaMa-ONNX)
  - **Important**: You **MUST** use `lama_fp32.onnx`.

### Depth Anything V2
- **Original Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- **Official Repository**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Model Source**: [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small/tree/main/onnx)
  - *Recommended File*: `depth_anything_v2_small.onnx` (Renamed from `model.onnx`).

### Segment Anything (SAM)

#### MobileSAM
- **Original Repository**: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- **Model Source**: [Acly/MobileSAM via HuggingFace](https://huggingface.co/Acly/MobileSAM/tree/main)

#### SAM 2 (Segment Anything Model 2)
- **Original Repository**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **Model Source**: [vietanhdev/segment-anything-2-onnx-models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models)

### Real-ESRGAN (Super Resolution)
- **Original Paper**: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- **Original Repository**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Model Source**: [qualcomm/Real-ESRGAN-x4plus via HuggingFace](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/tree/83db0da6e1b4969f85fe60ee713bd2c2b3160c23)
  - *Recommended File*: `Real-ESRGAN-x4plus.onnx`

### RMBG-1.4 (Background Removal)
- **Created By**: [BRIA AI](https://huggingface.co/briaai)
- **Model Source**: [briaai/RMBG-1.4 via HuggingFace](https://huggingface.co/briaai/RMBG-1.4)
  - *License Note*: RMBG-1.4 is released under a **Creative Commons license for non-commercial use**. For commercial use, please consult BRIA AI's licensing terms.

### Disclaimer
This project is an independent implementation for testing and educational purposes.