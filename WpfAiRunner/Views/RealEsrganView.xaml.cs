using OnnxEngines.Upscaling;
using OnnxEngines.Utils;
using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views;

public partial class RealEsrganView : UserControl, IDisposable
{
    private readonly RealEsrganEngine _engine = new();
    private byte[]? _inputBytes;
    private string? _currentModelPath;

    public RealEsrganView() => InitializeComponent();
    public void Dispose() => _engine.Dispose();

    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
#if DEBUG
        if (string.IsNullOrEmpty(_currentModelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("Real-ESRGAN-x4plus.onnx");
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
        var dlg = new OpenFileDialog { Filter = "ONNX Model|*.onnx" };
        if (dlg.ShowDialog() != true) return;

        _currentModelPath = dlg.FileName;
        await ReloadModel();
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        if (string.IsNullOrEmpty(_currentModelPath)) return;

        SetBusy(true, "Loading Model..."); // 여기서는 Indeterminate(뱅글뱅글) 모드
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            await Task.Run(() => _engine.LoadModel(_currentModelPath, useGpu));

            TxtStatus.Text = $"Model Loaded ({_engine.DeviceMode})";
            if (useGpu && _engine.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
            }

            BtnOpenImage.IsEnabled = true;
            if (_inputBytes != null) BtnUpscale.IsEnabled = true;
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

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg" };
        if (dlg.ShowDialog() != true) return;

        var bmp = new BitmapImage(new Uri(dlg.FileName));
        ImgInput.Source = bmp;

        using var ms = new MemoryStream();
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        enc.Save(ms);
        _inputBytes = ms.ToArray();

        BtnUpscale.IsEnabled = true;
        BtnSave.IsEnabled = false;
        ImgOutput.Source = null;
        TxtStatus.Text = "Image Loaded.";
    }

    private async void BtnUpscale_Click(object sender, RoutedEventArgs e)
    {
        if (_inputBytes == null) return;

        SetBusy(true, "Upscaling... 0%");

        // 작업 모드로 전환: 퍼센트 표시를 위해 Indeterminate 끄기
        PbarStatus.IsIndeterminate = false;
        PbarStatus.Value = 0;

        var progress = new Progress<double>(p =>
        {
            PbarStatus.Value = p * 100;
            TxtStatus.Text = $"Upscaling... {(int)(p * 100)}%";
        });

        try
        {
            byte[] resultBytes = await Task.Run(() => _engine.Upscale(_inputBytes, progress));

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

    private void BtnSave_Click(object sender, RoutedEventArgs e)
    {
        if (ImgOutput.Source is not BitmapSource resultBmp) return;

        var dlg = new SaveFileDialog { Filter = "PNG Image|*.png", FileName = "upscaled.png" };
        if (dlg.ShowDialog() != true) return;

        try
        {
            using var fileStream = new FileStream(dlg.FileName, FileMode.Create);
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(resultBmp));
            encoder.Save(fileStream);
            MessageBox.Show("Saved!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to save: {ex.Message}");
        }
    }

    private void SetBusy(bool busy, string? statusMsg = null)
    {
        PbarStatus.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;

        // 기본값은 Indeterminate (로딩용)
        // Upscale 버튼 클릭 시에만 수동으로 false로 바꿉니다.
        if (busy) PbarStatus.IsIndeterminate = true;

        BtnLoadModel.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && !string.IsNullOrEmpty(_currentModelPath);
        BtnUpscale.IsEnabled = !busy && _inputBytes != null && !string.IsNullOrEmpty(_currentModelPath);
        BtnSave.IsEnabled = !busy && ImgOutput.Source != null;

        if (statusMsg != null)
        {
            TxtStatus.Text = statusMsg;
        }
    }
}