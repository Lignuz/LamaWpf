using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using OnnxEngines.Utils;
using OpenCvSharp;

namespace OnnxEngines.Depth;

public class DepthEstimator : BaseOnnxEngine
{
    private const int ModelSize = 518;

    private Tensor<float>? _lastOutputTensor;
    private int _lastOrigW, _lastOrigH;

    private Mat? _lastInputMat;
    private Mat? _lastDepthMat;

    public DepthEstimator(string modelPath, bool useGpu = false) : base(modelPath, useGpu) { }

    protected override void OnWarmup()
    {
        if (_session == null) return;

        try
        {
            var dummyTensor = new DenseTensor<float>(new[] { 1, 3, ModelSize, ModelSize });
            string inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, dummyTensor) });
        }
        catch { }
    }

    // 1단계: 추론 수행 및 데이터 캡처
    public void RunInference(byte[] imageBytes)
    {
        if (_session == null) throw new System.InvalidOperationException("Model not loaded.");

        // SkiaSharp 기반 이미지 로드
        using var src = SKBitmap.Decode(imageBytes).Copy(SKColorType.Rgba8888);
        _lastOrigW = src.Width;
        _lastOrigH = src.Height;

        // TensorHelper를 이용한 전처리
        var inputTensor = src.ToTensor(ModelSize, ModelSize);

        // 리포커싱용 Mat 데이터 보관
        _lastInputMat?.Dispose();
        _lastInputMat = Cv2.ImDecode(imageBytes, ImreadModes.Color);

        var inputs = new List<NamedOnnxValue>();
        string inputName = _session.InputMetadata.Keys.First();
        inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));

        using var results = _session.Run(inputs);
        var outputRaw = results.First().AsTensor<float>();

        // 기존 방식의 텐서 캐싱
        _lastOutputTensor = outputRaw.ToDenseTensor();

        // 리포커싱용 float Mat 데이터 생성
        _lastDepthMat?.Dispose();
        _lastDepthMat = new Mat(ModelSize, ModelSize, MatType.CV_32FC1);
        for (int y = 0; y < ModelSize; y++)
            for (int x = 0; x < ModelSize; x++)
                _lastDepthMat.Set(y, x, _lastOutputTensor[0, y, x]);
    }

    // 2단계: 저장된 결과로 스타일 적용
    public byte[] GetDepthMap(ColormapStyle style)
    {
        if (_lastOutputTensor == null)
            throw new InvalidOperationException("Inference has not been run yet.");

        // 캐시된 텐서를 사용하여 이미지 생성
        using var outputImg = TensorToColorMap(_lastOutputTensor, ModelSize, ModelSize, style);

        // SkiaSharp 기반 원본 크기 복원
        using var resizedImg = outputImg.Resize(new SKImageInfo(_lastOrigW, _lastOrigH), new SKSamplingOptions(SKCubicResampler.Mitchell));
        using var data = resizedImg.Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    private SKBitmap TensorToColorMap(Tensor<float> tensor, int w, int h, ColormapStyle style)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (var val in tensor)
        {
            if (val < min) min = val;
            if (val > max) max = val;
        }

        float range = max - min;
        if (range < 0.00001f) range = 1f;

        var img = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Premul);
        Span<byte> pixels = img.GetPixelSpan();
        int bytesPerPixel = img.BytesPerPixel;

        for (int y = 0; y < h; y++)
        {
            int rowOffset = y * img.RowBytes;
            for (int x = 0; x < w; x++)
            {
                float val = tensor[0, y, x];
                float norm = (val - min) / range;
                SKColor color = ColorMapper.GetColor(norm, style);

                int offset = rowOffset + (x * bytesPerPixel);
                pixels[offset] = color.Red;
                pixels[offset + 1] = color.Green;
                pixels[offset + 2] = color.Blue;
                pixels[offset + 3] = 255;
            }
        }
        return img;
    }

    // 고성능 리포커싱 (실시간 업데이트 대응용)
    public byte[] RenderRefocus(double relX, double relY, float blurStrength)
    {
        if (_lastInputMat == null || _lastDepthMat == null) return Array.Empty<byte>();

        int fx = (int)Math.Clamp(relX * _lastDepthMat.Width, 0, _lastDepthMat.Width - 1);
        int fy = (int)Math.Clamp(relY * _lastDepthMat.Height, 0, _lastDepthMat.Height - 1);
        float focusDepth = _lastDepthMat.At<float>(fy, fx);

        using var fullDepth = new Mat();
        Cv2.Resize(_lastDepthMat, fullDepth, _lastInputMat.Size());

        using var blurMat = new Mat();
        int kSize = ((int)blurStrength * 4) + 1;
        if (kSize <= 1) return _lastInputMat.ToBytes(".png");

        Cv2.GaussianBlur(_lastInputMat, blurMat, new Size(kSize, kSize), 0);
        using var result = new Mat(_lastInputMat.Size(), _lastInputMat.Type());

        // 슬라이더 강도에 따른 감도 및 전이 부드러움 보정
        float sensitivity = 0.2f + (blurStrength / 30f) * 0.4f;

        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Cols; j++)
            {
                float diff = Math.Abs(fullDepth.At<float>(i, j) - focusDepth);
                float weight = Math.Min(diff * sensitivity, 1.0f);
                weight = weight * weight; // 비선형 가중치로 경계면 최적화

                var s = _lastInputMat.At<Vec3b>(i, j);
                var b = blurMat.At<Vec3b>(i, j);
                result.Set(i, j, new Vec3b(
                    (byte)(s.Item0 * (1 - weight) + b.Item0 * weight),
                    (byte)(s.Item1 * (1 - weight) + b.Item1 * weight),
                    (byte)(s.Item2 * (1 - weight) + b.Item2 * weight)));
            }
        }
        return result.ToBytes(".png");
    }

    // 배경 흑백 효과: 초점 외 영역의 채도를 단계별로 낮춤
    public byte[] RenderColorIsolation(double relX, double relY, float isolationStrength)
    {
        if (_lastInputMat == null || _lastDepthMat == null) return Array.Empty<byte>();

        // 1. 클릭 지점의 깊이값 추출
        int fx = (int)Math.Clamp(relX * _lastDepthMat.Width, 0, _lastDepthMat.Width - 1);
        int fy = (int)Math.Clamp(relY * _lastDepthMat.Height, 0, _lastDepthMat.Height - 1);
        float focusDepth = _lastDepthMat.At<float>(fy, fx);

        using var fullDepth = new Mat();
        Cv2.Resize(_lastDepthMat, fullDepth, _lastInputMat.Size());

        // 2. 전체 이미지를 흑백으로 변환한 레이어 생성
        using var grayMat = new Mat();
        Cv2.CvtColor(_lastInputMat, grayMat, ColorConversionCodes.BGR2GRAY);
        using var grayBgrMat = new Mat();
        Cv2.CvtColor(grayMat, grayBgrMat, ColorConversionCodes.GRAY2BGR);

        using var result = new Mat(_lastInputMat.Size(), _lastInputMat.Type());

        // 3. 슬라이더 강도에 따른 감도 설정
        float sensitivity = 0.3f + (isolationStrength / 30f) * 0.7f;

        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Cols; j++)
            {
                float diff = Math.Abs(fullDepth.At<float>(i, j) - focusDepth);
                // 깊이 차이가 클수록 가중치가 커짐 (흑백 비중 증가)
                float weight = Math.Min(diff * sensitivity, 1.0f);
                weight = weight * weight; // 전이를 더 부드럽게 처리

                var colorPix = _lastInputMat.At<Vec3b>(i, j);
                var grayPix = grayBgrMat.At<Vec3b>(i, j);

                // 컬러와 흑백을 가중치에 따라 합성
                result.Set(i, j, new Vec3b(
                    (byte)(colorPix.Item0 * (1 - weight) + grayPix.Item0 * weight),
                    (byte)(colorPix.Item1 * (1 - weight) + grayPix.Item1 * weight),
                    (byte)(colorPix.Item2 * (1 - weight) + grayPix.Item2 * weight)));
            }
        }
        return result.ToBytes(".png");
    }

    public override void Dispose()
    {
        base.Dispose(); // BaseOnnxEngine 자원 해제
        _lastInputMat?.Dispose();
        _lastDepthMat?.Dispose();
    }
}