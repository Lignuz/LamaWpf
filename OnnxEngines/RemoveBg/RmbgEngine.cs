using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Rmbg;

public class RmbgEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "CPU";
    private const int ModelSize = 1024;

    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);

        if (DeviceMode == "GPU")
        {
            try
            {
                var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
                string inputName = _session.InputMetadata.Keys.First();
                using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
            }
            catch { }
        }
    }

    public byte[] RemoveBackground(byte[] imageBytes, float threshold = 0.0f, Rgba32? bgColor = null)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        using var srcImage = Image.Load<Rgba32>(imageBytes);
        int originalW = srcImage.Width;
        int originalH = srcImage.Height;

        // 원본 srcImage가 변경되지 않도록 Clone()을 사용해야 합니다!
        // ToTensor는 내부적으로 리사이즈를 수행하므로, 원본에 영향을 주지 않으려면 복사본을 넘겨야 합니다.
        using var tempImage = srcImage.Clone();
        var inputTensor = tempImage.ToTensor(ModelSize, ModelSize);

        // 추론
        var inputName = _session.InputMetadata.Keys.First();
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
        var outputTensor = results.First().AsTensor<float>();

        // 마스크 생성
        using var maskImage = new Image<L8>(ModelSize, ModelSize);
        maskImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    float val = outputTensor[0, 0, y, x];
                    if (val < threshold) val = 0;
                    row[x] = new L8((byte)Math.Clamp(val * 255, 0, 255));
                }
            }
        });

        // 마스크를 원본 크기로 리사이즈
        maskImage.Mutate(x => x.Resize(originalW, originalH));

        // 결과 합성
        using var resultImage = new Image<Rgba32>(originalW, originalH);

        // 이제 srcImage가 원본 크기를 유지하고 있으므로 정상적으로 합성됩니다.
        resultImage.ProcessPixelRows(srcImage, maskImage, (resAccessor, srcAccessor, maskAccessor) =>
        {
            for (int y = 0; y < originalH; y++)
            {
                var resRow = resAccessor.GetRowSpan(y);
                var srcRow = srcAccessor.GetRowSpan(y);
                var maskRow = maskAccessor.GetRowSpan(y);

                for (int x = 0; x < originalW; x++)
                {
                    var srcPixel = srcRow[x];
                    float alpha = maskRow[x].PackedValue / 255.0f;

                    if (bgColor.HasValue)
                    {
                        var bg = bgColor.Value;
                        byte r = (byte)(srcPixel.R * alpha + bg.R * (1 - alpha));
                        byte g = (byte)(srcPixel.G * alpha + bg.G * (1 - alpha));
                        byte b = (byte)(srcPixel.B * alpha + bg.B * (1 - alpha));
                        resRow[x] = new Rgba32(r, g, b, 255);
                    }
                    else
                    {
                        srcPixel.A = (byte)(alpha * 255);
                        resRow[x] = srcPixel;
                    }
                }
            }
        });

        using var ms = new MemoryStream();
        resultImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose() => _session?.Dispose();
}