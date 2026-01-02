using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OnnxEngines.Utils;

namespace OnnxEngines.Colorization;

public class ColorizationEngine : IDisposable
{
    private InferenceSession? _session;
    public string DeviceMode { get; private set; } = "None";

    // DDColor 모델 입력 크기 (512x512 고정)
    private const int ModelInputSize = 512;

    // 1. LoadModel 메서드
    public void LoadModel(string modelPath, bool useGpu)
    {
        _session?.Dispose();
        (_session, DeviceMode) = OnnxHelper.LoadSession(modelPath, useGpu);
    }

    // 2. Process 메서드
    public byte[] Process(byte[] imageBytes)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded.");

        // 원본 이미지 로드
        using var originalImage = Image.Load<Rgba32>(imageBytes);
        int origW = originalImage.Width;
        int origH = originalImage.Height;

        // ---------------------------------------------------------
        // 단계 1: 모델 입력 준비 (512x512, Lab의 L채널 기반 회색조)
        // ---------------------------------------------------------
        using var inputImage = originalImage.Clone(x => x.Resize(ModelInputSize, ModelInputSize));

        var inputTensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });

        inputImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    float r = row[x].R / 255.0f;
                    float g = row[x].G / 255.0f;
                    float b = row[x].B / 255.0f;

                    // RGB -> Lab 변환 후 L 채널만 추출
                    RgbToLab(r, g, b, out float L, out _, out _);

                    // L 채널만 있는(a=0, b=0) 회색조 RGB로 다시 변환
                    // DDColor는 이렇게 '순수한 밝기'만 남긴 RGB 입력을 기대합니다.
                    LabToRgb(L, 0f, 0f, out float grayR, out float grayG, out float grayB);

                    // 정규화 없이 0.0 ~ 1.0 값 그대로 입력
                    inputTensor[0, 0, y, x] = grayR;
                    inputTensor[0, 1, y, x] = grayG;
                    inputTensor[0, 2, y, x] = grayB;
                }
            }
        });

        // ---------------------------------------------------------
        // 단계 2: 추론 (Inference)
        // ---------------------------------------------------------
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor) };
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // 출력 레이아웃 확인 (NCHW vs NHWC)
        bool isNchw = outputTensor.Dimensions[1] == 2;

        // ---------------------------------------------------------
        // 단계 3: 결과 합성 (블렌딩 & 원본 해상도 복원)
        // ---------------------------------------------------------
        using var outputImage = new Image<Rgba32>(origW, origH);

        // 원본과 출력 이미지를 동시에 순회
        outputImage.ProcessPixelRows(originalImage, (targetAccessor, sourceAccessor) =>
        {
            for (int y = 0; y < targetAccessor.Height; y++)
            {
                var targetRow = targetAccessor.GetRowSpan(y);
                var sourceRow = sourceAccessor.GetRowSpan(y);

                for (int x = 0; x < targetAccessor.Width; x++)
                {
                    // (A) 원본 픽셀에서 L(밝기) 추출 - 화질 보존
                    Rgba32 pixel = sourceRow[x];
                    float r = pixel.R / 255.0f;
                    float g = pixel.G / 255.0f;
                    float b = pixel.B / 255.0f;

                    RgbToLab(r, g, b, out float origL, out _, out _);

                    // (B) 모델 출력에서 a, b(색상) 추출 (좌표 매핑)
                    // 원본 좌표(x,y)를 모델 좌표(512,512)로 변환
                    int modelX = (int)((float)x / origW * ModelInputSize);
                    int modelY = (int)((float)y / origH * ModelInputSize);
                    modelX = Math.Clamp(modelX, 0, ModelInputSize - 1);
                    modelY = Math.Clamp(modelY, 0, ModelInputSize - 1);

                    float predA, predB;
                    if (isNchw)
                    {
                        predA = outputTensor[0, 0, modelY, modelX];
                        predB = outputTensor[0, 1, modelY, modelX];
                    }
                    else
                    {
                        predA = outputTensor[0, modelY, modelX, 0];
                        predB = outputTensor[0, modelY, modelX, 1];
                    }

                    // (C) Lab -> RGB 변환 및 저장
                    // 모델 출력 그대로 사용 (ColorScale 제거 권장사항 반영)
                    // 만약 색이 너무 연하면 predA * 1.2f 정도로 살짝만 키우세요.
                    LabToRgb(origL, predA, predB, out float outR, out float outG, out float outB);

                    targetRow[x] = new Rgba32(
                        (byte)Math.Clamp(outR * 255f, 0, 255),
                        (byte)Math.Clamp(outG * 255f, 0, 255),
                        (byte)Math.Clamp(outB * 255f, 0, 255)
                    );
                }
            }
        });

        using var ms = new MemoryStream();
        outputImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    public void Dispose() => _session?.Dispose();


    // ================================
    // OpenCV 스타일 색공간 변환 함수들
    // ================================

    private static void RgbToLab(float r, float g, float b, out float L, out float a, out float bb)
    {
        // sRGB -> Linear RGB
        r = SrgbToLinear(r);
        g = SrgbToLinear(g);
        b = SrgbToLinear(b);

        // Linear RGB -> XYZ (D65)
        float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
        float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
        float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

        // XYZ -> Lab
        const float Xn = 0.95047f;
        const float Yn = 1.00000f;
        const float Zn = 1.08883f;

        float fx = Fxyz(X / Xn);
        float fy = Fxyz(Y / Yn);
        float fz = Fxyz(Z / Zn);

        L = 116f * fy - 16f;
        a = 500f * (fx - fy);
        bb = 200f * (fy - fz);
    }

    private static void LabToRgb(float L, float a, float bb, out float r, out float g, out float b)
    {
        // Lab -> XYZ
        float fy = (L + 16f) / 116f;
        float fx = fy + (a / 500f);
        float fz = fy - (bb / 200f);

        const float Xn = 0.95047f;
        const float Yn = 1.00000f;
        const float Zn = 1.08883f;

        float X = Xn * Finv(fx);
        float Y = Yn * Finv(fy);
        float Z = Zn * Finv(fz);

        // XYZ -> Linear RGB
        float rl = X * 3.2404542f + Y * -1.5371385f + Z * -0.4985314f;
        float gl = X * -0.9692660f + Y * 1.8760108f + Z * 0.0415560f;
        float bl = X * 0.0556434f + Y * -0.2040259f + Z * 1.0572252f;

        // Linear RGB -> sRGB
        r = LinearToSrgb(rl);
        g = LinearToSrgb(gl);
        b = LinearToSrgb(bl);

        // Clamp
        r = Math.Clamp(r, 0f, 1f);
        g = Math.Clamp(g, 0f, 1f);
        b = Math.Clamp(b, 0f, 1f);
    }

    private static float SrgbToLinear(float c)
    {
        if (c <= 0.04045f) return c / 12.92f;
        return MathF.Pow((c + 0.055f) / 1.055f, 2.4f);
    }

    private static float LinearToSrgb(float c)
    {
        if (c <= 0.0031308f) return 12.92f * c;
        return 1.055f * MathF.Pow(c, 1f / 2.4f) - 0.055f;
    }

    private static float Fxyz(float t)
    {
        const float delta = 6f / 29f;
        const float delta3 = delta * delta * delta;
        if (t > delta3) return MathF.Pow(t, 1f / 3f);
        return (t / (3f * delta * delta)) + (4f / 29f);
    }

    private static float Finv(float ft)
    {
        const float delta = 6f / 29f;
        if (ft > delta) return ft * ft * ft;
        return 3f * delta * delta * (ft - 4f / 29f);
    }
}