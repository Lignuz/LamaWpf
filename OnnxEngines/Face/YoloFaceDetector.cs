using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Face;

public class YoloFaceDetector : IFaceDetector
{
    private readonly InferenceSession _session;
    public string DeviceMode { get; private set; } = "CPU";

    // YOLOv8n-Face 입력 크기
    private const int InputSize = 640;

    public YoloFaceDetector(string modelPath, bool useGpu = false)
    {
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);
    }

    public List<Rectangle> DetectFaces(byte[] imageBytes, float confThreshold = 0.5f)
    {
        using var image = Image.Load<Rgba32>(imageBytes);
        int origW = image.Width;
        int origH = image.Height;

        // 1. 전처리 (Resize 640x640, Normalize 0..1)
        using var resized = image.Clone(x => x.Resize(InputSize, InputSize));
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

        resized.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    inputTensor[0, 0, y, x] = row[x].R / 255.0f;
                    inputTensor[0, 1, y, x] = row[x].G / 255.0f;
                    inputTensor[0, 2, y, x] = row[x].B / 255.0f;
                }
            }
        });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", inputTensor)
        };

        // 2. 추론
        using var results = _session.Run(inputs);
        // Output: [1, 5, 8400] (cx, cy, w, h, score)
        var output = results.First().AsTensor<float>();

        var candidates = new List<(Rectangle Rect, float Score)>();
        int anchors = output.Dimensions[2]; // 8400

        for (int i = 0; i < anchors; i++)
        {
            float score = output[0, 4, i]; // Score
            if (score > confThreshold)
            {
                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                // Scale 복원
                float x = (cx - w / 2) * (origW / (float)InputSize);
                float y = (cy - h / 2) * (origH / (float)InputSize);
                float width = w * (origW / (float)InputSize);
                float height = h * (origH / (float)InputSize);

                candidates.Add((new Rectangle((int)x, (int)y, (int)width, (int)height), score));
            }
        }

        return NMS(candidates);
    }

    private List<Rectangle> NMS(List<(Rectangle Rect, float Score)> boxes, float iouThreshold = 0.45f)
    {
        var result = new List<Rectangle>();
        var sorted = boxes.OrderByDescending(x => x.Score).ToList();

        while (sorted.Count > 0)
        {
            var current = sorted[0];
            result.Add(current.Rect);
            sorted.RemoveAt(0);
            sorted.RemoveAll(other => CalculateIoU(current.Rect, other.Rect) > iouThreshold);
        }
        return result;
    }

    private float CalculateIoU(Rectangle r1, Rectangle r2)
    {
        var intersect = Rectangle.Intersect(r1, r2);
        float intersectionArea = intersect.Width * intersect.Height;
        if (intersect.Width <= 0 || intersect.Height <= 0) return 0f;
        float unionArea = (r1.Width * r1.Height) + (r2.Width * r2.Height) - intersectionArea;
        return intersectionArea / unionArea;
    }

    // 그리기 함수 (FaceDetector와 동일 로직)
    public byte[] ApplyBlur(byte[] imageBytes, List<Rectangle> faces, int blurSigma = 15)
    {
        using var image = Image.Load<Rgba32>(imageBytes);
        foreach (var face in faces)
        {
            var roi = Rectangle.Intersect(face, image.Bounds);
            if (roi.Width <= 1 || roi.Height <= 1) continue;
            int safeSigma = Math.Min(blurSigma, Math.Min(roi.Width, roi.Height) / 4);
            if (safeSigma < 1) safeSigma = 1;

            image.Mutate(ctx =>
            {
                using var part = image.Clone(c => c.Crop(roi).GaussianBlur(safeSigma));
                ctx.DrawImage(part, new Point(roi.X, roi.Y), 1.0f);
            });
        }
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    public byte[] DrawBoundingBoxes(byte[] imageBytes, List<Rectangle> faces, float thickness = 3)
    {
        using var image = Image.Load<Rgba32>(imageBytes);
        var pen = Pens.Solid(Color.Red, thickness);
        image.Mutate(ctx => { foreach (var face in faces) ctx.Draw(pen, face); });
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose() => _session?.Dispose();
}