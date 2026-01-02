using Microsoft.Win32;
using OnnxEngines.Face;
using OnnxEngines.Utils;
using SixLabors.ImageSharp;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views;

public partial class FaceView : UserControl, IDisposable
{
    private IFaceDetector? _detector;
    private byte[]? _inputBytes;
    private List<Rectangle>? _cachedFaces;

    private string? _rfbPath;
    private string? _yolo8Path;
    private string? _yolo11Path;

    public FaceView() => InitializeComponent();
    public void Dispose() => _detector?.Dispose();

    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
#if DEBUG
        _rfbPath = OnnxHelper.FindModelInDebug("version-RFB-320.onnx");
        _yolo8Path = OnnxHelper.FindModelInDebug("yolov8n-face.onnx");
        _yolo11Path = OnnxHelper.FindModelInDebug("yolov11n-face.onnx"); // [추가]
        await ReloadModel();
#endif
    }

    private async void CboModelSelect_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        SetBusy(true, "Switching Model...");
        try
        {
            _detector?.Dispose();
            _detector = null;

            bool useGpu = ChkUseGpu.IsChecked == true;
            int selectedIndex = CboModelSelect.SelectedIndex;
            string modelName = selectedIndex switch
            {
                0 => "RFB-320",
                1 => "YOLOv8",
                2 => "YOLOv11",
                _ => "Unknown"
            };

            await Task.Run(() =>
            {
                if (selectedIndex == 0) // RFB-320
                {
                    if (!string.IsNullOrEmpty(_rfbPath) && File.Exists(_rfbPath))
                        _detector = new FaceDetector(_rfbPath, useGpu);
                }
                else if (selectedIndex == 1) // YOLOv8
                {
                    if (!string.IsNullOrEmpty(_yolo8Path) && File.Exists(_yolo8Path))
                        _detector = new YoloFaceDetector(_yolo8Path, useGpu);
                }
                else // YOLOv11 (YOLOv8과 동일한 엔진 사용)
                {
                    if (!string.IsNullOrEmpty(_yolo11Path) && File.Exists(_yolo11Path))
                        _detector = new YoloFaceDetector(_yolo11Path, useGpu);
                }
            });

            if (_detector != null)
            {
                TxtStatus.Text = $"{modelName} Loaded ({_detector.DeviceMode})";
                if (useGpu && _detector.DeviceMode.Contains("CPU")) ChkUseGpu.IsChecked = false;

                BtnOpenImage.IsEnabled = true;

                // 모델 변경 시 이미지가 있다면 즉시 재추론
                if (_inputBytes != null)
                {
                    BtnRun.IsEnabled = true;
                    _cachedFaces = null;
                    ImgOutput.Source = null;
                    await RunDetection();
                }
            }
            else
            {
                TxtStatus.Text = "Model file not found.";
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Load Error: {ex.Message}");
            TxtStatus.Text = "Load Failed";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg" };
        if (dlg.ShowDialog() == true)
        {
            var bmp = new BitmapImage(new Uri(dlg.FileName));
            ImgInput.Source = bmp;

            using var ms = new MemoryStream();
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            enc.Save(ms);
            _inputBytes = ms.ToArray();

            _cachedFaces = null;
            ImgOutput.Source = null;

            BtnRun.IsEnabled = (_detector != null);
            BtnSave.IsEnabled = false;
            TxtStatus.Text = "Image Loaded.";
        }
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        await RunDetection();
    }

    private async Task RunDetection()
    {
        if (_detector == null || _inputBytes == null) return;

        if (_cachedFaces != null)
        {
            await RenderResult();
            return;
        }

        SetBusy(true, "Detecting faces...");
        try
        {
            _cachedFaces = await Task.Run(() => _detector.DetectFaces(_inputBytes));

            if (_cachedFaces.Count == 0)
            {
                TxtStatus.Text = "No faces found (Original shown).";
            }

            await RenderResult();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            TxtStatus.Text = "Error.";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void ChkOption_Click(object sender, RoutedEventArgs e)
    {
        if (_cachedFaces != null && _cachedFaces.Count > 0)
        {
            await RenderResult();
        }
    }

    private async Task RenderResult()
    {
        if (_detector == null || _inputBytes == null || _cachedFaces == null) return;

        SetBusy(true, "Rendering...");
        try
        {
            bool applyBlur = ChkBlur.IsChecked == true;
            bool drawBox = ChkBox.IsChecked == true;

            byte[] processedBytes = _inputBytes.ToArray();

            if (_cachedFaces.Count > 0)
            {
                await Task.Run(() =>
                {
                    if (applyBlur)
                        processedBytes = _detector.ApplyBlur(processedBytes, _cachedFaces);
                    if (drawBox)
                        processedBytes = _detector.DrawBoundingBoxes(processedBytes, _cachedFaces);
                });
            }

            using var ms = new MemoryStream(processedBytes);
            var resultBmp = new BitmapImage();
            resultBmp.BeginInit();
            resultBmp.StreamSource = ms;
            resultBmp.CacheOption = BitmapCacheOption.OnLoad;
            resultBmp.EndInit();
            resultBmp.Freeze();

            ImgOutput.Source = resultBmp;
            BtnSave.IsEnabled = true;

            if (_cachedFaces.Count > 0)
                TxtStatus.Text = $"Done. {_cachedFaces.Count} faces processed.";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void BtnSave_Click(object sender, RoutedEventArgs e)
    {
        if (ImgOutput.Source is not BitmapSource bmp) return;
        var dlg = new SaveFileDialog { Filter = "PNG|*.png", FileName = "result.png" };
        if (dlg.ShowDialog() == true)
        {
            using var stream = new FileStream(dlg.FileName, FileMode.Create);
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            enc.Save(stream);
        }
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        await ReloadModel();
    }

    private void SetBusy(bool busy, string? msg = null)
    {
        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        CboModelSelect.IsEnabled = !busy;

        BtnLoadModel.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _detector != null;
        BtnRun.IsEnabled = !busy && _detector != null && _inputBytes != null;
        ChkBlur.IsEnabled = !busy;
        ChkBox.IsEnabled = !busy;

        if (msg != null) TxtStatus.Text = msg;
    }
}