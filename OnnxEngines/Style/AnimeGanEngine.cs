using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Style;

public class AnimeGanEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "None";

    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);
    }

    public byte[] Process(byte[] imageBytes)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var image = Image.Load<Rgba32>(imageBytes);

        // 1. 크기 조정 (32의 배수)
        int w = image.Width - (image.Width % 32);
        int h = image.Height - (image.Height % 32);

        if (w != image.Width || h != image.Height)
        {
            image.Mutate(x => x.Resize(w, h));
        }

        // 2. 텐서 변환: NHWC 포맷 [1, Height, Width, Channels]
        var inputTensor = new DenseTensor<float>(new[] { 1, h, w, 3 });

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    // 채널이 마지막 차원 (NHWC)
                    inputTensor[0, y, x, 0] = (row[x].R / 127.5f) - 1.0f; // R
                    inputTensor[0, y, x, 1] = (row[x].G / 127.5f) - 1.0f; // G
                    inputTensor[0, y, x, 2] = (row[x].B / 127.5f) - 1.0f; // B
                }
            }
        });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
        };

        // 3. 추론
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // 4. 후처리: NHWC 포맷 읽기
        using var outputImage = new Image<Rgba32>(w, h);
        outputImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    // 출력 텐서도 [1, h, w, 3] 형태
                    float r = (outputTensor[0, y, x, 0] + 1.0f) * 127.5f;
                    float g = (outputTensor[0, y, x, 1] + 1.0f) * 127.5f;
                    float b = (outputTensor[0, y, x, 2] + 1.0f) * 127.5f;

                    row[x] = new Rgba32(
                        (byte)Math.Clamp(r, 0, 255),
                        (byte)Math.Clamp(g, 0, 255),
                        (byte)Math.Clamp(b, 0, 255)
                    );
                }
            }
        });

        using var ms = new MemoryStream();
        outputImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose() => _session?.Dispose();
}