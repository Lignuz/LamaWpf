using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Face;

public class FaceDetector : IDisposable
{
    private readonly InferenceSession _session;
    public string DeviceMode { get; private set; } = "CPU";

    private const int InputWidth = 320;
    private const int InputHeight = 240;

    public FaceDetector(string modelPath, bool useGpu = false)
    {
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);
    }

    public List<Rectangle> DetectFaces(byte[] imageBytes, float confThreshold = 0.7f)
    {
        using var image = Image.Load<Rgba32>(imageBytes);
        int origW = image.Width;
        int origH = image.Height;

        using var resized = image.Clone(x => x.Resize(InputWidth, InputHeight));
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });

        resized.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    inputTensor[0, 0, y, x] = (row[x].R - 127.0f) / 128.0f;
                    inputTensor[0, 1, y, x] = (row[x].G - 127.0f) / 128.0f;
                    inputTensor[0, 2, y, x] = (row[x].B - 127.0f) / 128.0f;
                }
            }
        });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
        };

        using var results = _session.Run(inputs);
        var confidences = results.First(x => x.Name == "scores").AsTensor<float>();
        var boxes = results.First(x => x.Name == "boxes").AsTensor<float>();

        // NMS를 위해 점수와 박스를 함께 저장
        var candidates = new List<(Rectangle Rect, float Score)>();
        int numAnchors = confidences.Dimensions[1];

        for (int i = 0; i < numAnchors; i++)
        {
            float score = confidences[0, i, 1];
            if (score > confThreshold)
            {
                float x = boxes[0, i, 0] * origW;
                float y = boxes[0, i, 1] * origH;
                float w = (boxes[0, i, 2] - boxes[0, i, 0]) * origW;
                float h = (boxes[0, i, 3] - boxes[0, i, 1]) * origH;

                candidates.Add((new Rectangle((int)x, (int)y, (int)w, (int)h), score));
            }
        }

        // NMS 적용하여 중복 제거 후 반환
        return NMS(candidates);
    }

    // Non-Maximum Suppression (중복 박스 제거)
    private List<Rectangle> NMS(List<(Rectangle Rect, float Score)> boxes, float iouThreshold = 0.3f)
    {
        var result = new List<Rectangle>();
        // 점수가 높은 순으로 정렬
        var sorted = boxes.OrderByDescending(x => x.Score).ToList();

        while (sorted.Count > 0)
        {
            // 가장 점수가 높은 박스 선택
            var current = sorted[0];
            result.Add(current.Rect);
            sorted.RemoveAt(0);

            // 선택된 박스와 많이 겹치는(IoU가 높은) 박스들은 제거
            sorted.RemoveAll(other => CalculateIoU(current.Rect, other.Rect) > iouThreshold);
        }

        return result;
    }

    // IoU (Intersection over Union) 계산
    private float CalculateIoU(Rectangle r1, Rectangle r2)
    {
        var intersect = Rectangle.Intersect(r1, r2);
        float intersectionArea = intersect.Width * intersect.Height;

        if (intersect.Width <= 0 || intersect.Height <= 0) return 0f;

        float unionArea = (r1.Width * r1.Height) + (r2.Width * r2.Height) - intersectionArea;
        return intersectionArea / unionArea;
    }

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

        image.Mutate(ctx =>
        {
            foreach (var face in faces)
            {
                ctx.Draw(pen, face);
            }
        });

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose() => _session?.Dispose();
}