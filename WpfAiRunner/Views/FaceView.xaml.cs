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
    private FaceDetector? _detector;
    private byte[]? _inputBytes;
    private string? _currentModelPath;
    private List<Rectangle>? _cachedFaces;

    public FaceView() => InitializeComponent();
    public void Dispose() => _detector?.Dispose();

    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
#if DEBUG
        if (_detector == null && string.IsNullOrEmpty(_currentModelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("version-RFB-320.onnx");
            if (debugPath != null)
            {
                _currentModelPath = debugPath;
                await ReloadModel();
            }
        }
#endif
    }

    private async void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX (*.onnx)|*.onnx" };
        if (dlg.ShowDialog() == true)
        {
            _currentModelPath = dlg.FileName;
            await ReloadModel();
        }
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(_currentModelPath))
            await ReloadModel();
    }

    private async Task ReloadModel()
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;

        SetBusy(true, "Loading Model...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _detector?.Dispose();
            _detector = await Task.Run(() => new FaceDetector(_currentModelPath!, useGpu));

            TxtStatus.Text = $"Model Loaded ({_detector.DeviceMode})";

            if (useGpu && _detector.DeviceMode.Contains("CPU"))
                ChkUseGpu.IsChecked = false;

            BtnOpenImage.IsEnabled = true;
            if (_inputBytes != null) BtnRun.IsEnabled = true;
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

            // 0개여도 return 하지 않고 진행
            if (_cachedFaces.Count == 0)
            {
                TxtStatus.Text = "No faces found.";
                // _cachedFaces는 빈 리스트 상태로 유지
            }

            // 결과 그리기 (0개면 원본 출력됨)
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
        // _cachedFaces가 null이면 실행 안 함 (빈 리스트면 실행 함)
        if (_detector == null || _inputBytes == null || _cachedFaces == null) return;

        SetBusy(true, "Rendering...");

        try
        {
            bool applyBlur = ChkBlur.IsChecked == true;
            bool drawBox = ChkBox.IsChecked == true;

            byte[] processedBytes = _inputBytes.ToArray();

            // 얼굴이 하나라도 있을 때만 처리
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

            // 상태 메시지 업데이트
            if (_cachedFaces.Count == 0)
                TxtStatus.Text = "No faces found (Original shown).";
            else
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
        var dlg = new SaveFileDialog { Filter = "PNG|*.png", FileName = "mosaic_result.png" };
        if (dlg.ShowDialog() == true)
        {
            using var stream = new FileStream(dlg.FileName, FileMode.Create);
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            enc.Save(stream);
        }
    }

    private void SetBusy(bool busy, string? msg = null)
    {
        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;

        BtnLoadModel.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _detector != null;
        BtnRun.IsEnabled = !busy && _detector != null && _inputBytes != null;
        ChkBlur.IsEnabled = !busy;
        ChkBox.IsEnabled = !busy;

        if (msg != null) TxtStatus.Text = msg;
    }
}