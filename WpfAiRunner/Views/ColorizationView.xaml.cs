using Microsoft.Win32;
using OnnxEngines.Colorization;
using OnnxEngines.Utils;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views;

public partial class ColorizationView : UserControl, IDisposable
{
    private ColorizationEngine? _engine;
    private byte[]? _inputBytes;
    private string? _currentModelPath;

    public ColorizationView() => InitializeComponent();
    public void Dispose() => _engine?.Dispose();

    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
#if DEBUG
        if (_engine == null)
        {
            //string? debugPath = OnnxHelper.FindModelInDebug("ddcolor_tiny_512.onnx");
            string? debugPath = OnnxHelper.FindModelInDebug("ddcolor.onnx");
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
        SetBusy(true, "Loading Model...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _engine?.Dispose();
            _engine = new ColorizationEngine();

            await Task.Run(() => _engine.LoadModel(_currentModelPath!, useGpu));

            TxtStatus.Text = $"DDColor Loaded ({_engine.DeviceMode})";
            if (useGpu && _engine.DeviceMode.Contains("CPU")) ChkUseGpu.IsChecked = false;

            BtnOpenImage.IsEnabled = true;
            if (_inputBytes != null) BtnRun.IsEnabled = true;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            TxtStatus.Text = "Load Failed";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp" };
        if (dlg.ShowDialog() == true)
        {
            var bmp = new BitmapImage(new Uri(dlg.FileName));
            ImgInput.Source = bmp;

            using var ms = new MemoryStream();
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            enc.Save(ms);
            _inputBytes = ms.ToArray();

            ImgOutput.Source = null;
            BtnRun.IsEnabled = (_engine != null);
            BtnSave.IsEnabled = false;
            TxtStatus.Text = "Image Loaded.";
        }
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_engine == null || _inputBytes == null) return;

        SetBusy(true, "Colorizing...");
        try
        {
            byte[] resultBytes = await Task.Run(() => _engine.Process(_inputBytes));

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
            TxtStatus.Text = "Error.";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void BtnSave_Click(object sender, RoutedEventArgs e)
    {
        if (ImgOutput.Source is not BitmapSource bmp) return;
        var dlg = new SaveFileDialog { Filter = "PNG|*.png", FileName = "colorized.png" };
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
        BtnOpenImage.IsEnabled = !busy && _engine != null;
        BtnRun.IsEnabled = !busy && _engine != null && _inputBytes != null;
        if (msg != null) TxtStatus.Text = msg;
    }
}