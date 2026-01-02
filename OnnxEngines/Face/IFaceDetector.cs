using SixLabors.ImageSharp;

namespace OnnxEngines.Face;

public interface IFaceDetector : IDisposable
{
    string DeviceMode { get; }

    // 얼굴 감지
    List<Rectangle> DetectFaces(byte[] imageBytes, float confThreshold = 0.5f);

    // 후처리 (블러 및 박스 그리기)
    byte[] ApplyBlur(byte[] imageBytes, List<Rectangle> faces, int blurSigma = 15);
    byte[] DrawBoundingBoxes(byte[] imageBytes, List<Rectangle> faces, float thickness = 3);
}