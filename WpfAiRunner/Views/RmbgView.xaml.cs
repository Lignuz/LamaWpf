using Microsoft.Win32;
using OnnxEngines.Rmbg;
using OnnxEngines.Utils;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views;

public partial class RmbgView : UserControl, IDisposable
{
    // 엔진 인스턴스
    private readonly RmbgEngine _engine = new();

    private byte[]? _inputBytes;
    private string? _currentModelPath;

    public RmbgView() => InitializeComponent();
    public void Dispose() => _engine.Dispose();

    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
#if DEBUG
        if (string.IsNullOrEmpty(_currentModelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("rmbg-1.4.onnx");
            if (debugPath != null)
            {
                _currentModelPath = debugPath;
                await ReloadModel();
            }
        }
#endif
    }

    // 1. 모델 로드 버튼
    private async void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX Model|*.onnx", Title = "Select RMBG-1.4 Model" };
        if (dlg.ShowDialog() != true) return;

        _currentModelPath = dlg.FileName;
        await ReloadModel();
    }

    // 2. GPU 토글
    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;
        await ReloadModel();
    }

    // [공통] 모델 재로딩 로직
    private async Task ReloadModel()
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;

        SetBusy(true, "Loading Model...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            await Task.Run(() => _engine.LoadModel(_currentModelPath, useGpu));

            TxtStatus.Text = $"Model Loaded ({_engine.DeviceMode})";
            BtnOpenImage.IsEnabled = true;

            if (_inputBytes != null) BtnRun.IsEnabled = true;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading model: {ex.Message}");
            TxtStatus.Text = "Load Failed";
            _currentModelPath = null;
        }
        finally
        {
            SetBusy(false);
        }
    }

    // 3. 이미지 열기
    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp" };
        if (dlg.ShowDialog() != true) return;

        try
        {
            var bmp = new BitmapImage(new Uri(dlg.FileName));
            ImgInput.Source = bmp;

            // 이미지를 바이트 배열로 변환
            using var ms = new MemoryStream();
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            enc.Save(ms);
            _inputBytes = ms.ToArray();

            BtnRun.IsEnabled = true;
            BtnSave.IsEnabled = false;
            ImgOutput.Source = null;
            TxtStatus.Text = "Image Loaded.";
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to load image: {ex.Message}");
        }
    }

    // 4. 배경 제거 실행
    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_inputBytes == null) return;

        SetBusy(true, "Processing...");

        try
        {
            // 1. UI 옵션 가져오기
            float threshold = (float)SldThreshold.Value;

            // 2. 배경색 결정
            Rgba32? bgColor = null;
            if (CboBackground.SelectedIndex > 0) // 0은 Transparent
            {
                bgColor = CboBackground.SelectedIndex switch
                {
                    1 => new Rgba32(255, 255, 255), // White
                    2 => new Rgba32(0, 0, 0),       // Black
                    3 => new Rgba32(0, 255, 0),     // Green
                    4 => new Rgba32(0, 0, 255),     // Blue
                    _ => null
                };
            }

            // 3. 엔진 실행 (옵션 전달)
            byte[] resultBytes = await Task.Run(() =>
                _engine.RemoveBackground(_inputBytes, threshold, bgColor));

            // 4. 결과 표시
            using var ms = new MemoryStream(resultBytes);
            var resultBmp = new BitmapImage();
            resultBmp.BeginInit();
            resultBmp.StreamSource = ms;
            resultBmp.CacheOption = BitmapCacheOption.OnLoad;
            resultBmp.EndInit();
            resultBmp.Freeze();

            ImgOutput.Source = resultBmp;
            BtnSave.IsEnabled = true;
            TxtStatus.Text = "Done.";
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            TxtStatus.Text = "Failed.";
        }
        finally
        {
            SetBusy(false);
        }
    }

    // 5. 결과 저장 (PNG)
    private void BtnSave_Click(object sender, RoutedEventArgs e)
    {
        if (ImgOutput.Source is not BitmapSource resultBmp) return;

        var dlg = new SaveFileDialog { Filter = "PNG Image|*.png", FileName = "rmbg_result.png" };
        if (dlg.ShowDialog() != true) return;

        try
        {
            using var fileStream = new FileStream(dlg.FileName, FileMode.Create);
            var encoder = new PngBitmapEncoder(); // 투명도 유지를 위해 반드시 PNG 사용
            encoder.Frames.Add(BitmapFrame.Create(resultBmp));
            encoder.Save(fileStream);
            MessageBox.Show("Saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to save: {ex.Message}");
        }
    }

    // 6. UI 상태 제어
    private void SetBusy(bool busy, string? statusMsg = null)
    {
        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;

        BtnLoadModel.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && !string.IsNullOrEmpty(_currentModelPath);
        BtnRun.IsEnabled = !busy && _inputBytes != null && !string.IsNullOrEmpty(_currentModelPath);
        BtnSave.IsEnabled = !busy && ImgOutput.Source != null;

        // 슬라이더 등도 잠그고 싶다면 추가
        SldThreshold.IsEnabled = !busy;
        CboBackground.IsEnabled = !busy;

        if (statusMsg != null)
        {
            TxtStatus.Text = statusMsg;
        }
    }
}