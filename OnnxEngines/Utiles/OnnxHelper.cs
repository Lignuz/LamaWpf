using Microsoft.ML.OnnxRuntime;

namespace OnnxEngines.Utils;

public static class OnnxHelper
{
    public static (InferenceSession Session, string DeviceMode) LoadSession(string modelPath, bool useGpu)
    {
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC
        };

        string deviceMode = "CPU";

        if (useGpu)
        {
            try
            {
                so.AppendExecutionProvider_CUDA(0);
                deviceMode = "GPU";
            }
            catch
            {
                deviceMode = "CPU (Fallback)";
                System.Diagnostics.Debug.WriteLine("GPU Load Failed. Fallback to CPU.");
            }
        }

        return (new InferenceSession(modelPath, so), deviceMode);
    }

    /// <summary>
    /// [Debug 모드 전용] x64 Debug 빌드 경로 기준으로 고정된 상대 경로를 사용하여 모델을 찾습니다.
    /// (bin/x64/Debug/net8.0-windows -> ../../../../../models)
    /// </summary>
    public static string? FindModelInDebug(string filename)
    {
#if DEBUG
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;

        // 현재 위치: .../WpfAiRunner/bin/x64/Debug/net8.0-windows/
        // 목표 위치: .../models/
        string fixedPath = Path.Combine(baseDir, "..", "..", "..", "..", "..", "models", filename);

        string fullPath = Path.GetFullPath(fixedPath); // 경로 정규화
        if (File.Exists(fullPath))
        {
            return fullPath;
        }
#endif
        return null;
    }
}