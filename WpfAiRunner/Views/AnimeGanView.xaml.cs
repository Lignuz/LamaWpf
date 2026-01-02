using Microsoft.Win32;
using OnnxEngines.Style;
using OnnxEngines.Utils;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views;

public partial class AnimeGanView : UserControl, IDisposable
{
    private AnimeGanEngine? _engine;
    private byte[]? _inputBytes;

    // 모델 경로들
    private string? _hayaoPath;
    private string? _shinkaiPath;
    private string? _paprikaPath;

    public AnimeGanView() => InitializeComponent();
    public void Dispose() => _engine?.Dispose();

    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
#if DEBUG
        // 모델 경로 찾기
        _hayaoPath = OnnxHelper.FindModelInDebug("AnimeGANv2_Hayao.onnx");
        _shinkaiPath = OnnxHelper.FindModelInDebug("AnimeGANv2_Shinkai.onnx");
        _paprikaPath = OnnxHelper.FindModelInDebug("AnimeGANv2_Paprika.onnx");

        await ReloadModel();
#endif
    }

    private async void CboStyle_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        await ReloadModel();
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        SetBusy(true, "Loading Model...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _engine?.Dispose();
            _engine = new AnimeGanEngine();

            int index = CboStyle.SelectedIndex;
            string? targetPath = index switch
            {
                0 => _hayaoPath,
                1 => _shinkaiPath,
                2 => _paprikaPath,
                _ => _hayaoPath
            };
            string styleName = index switch { 0 => "Hayao", 1 => "Shinkai", 2 => "Paprika", _ => "" };

            if (string.IsNullOrEmpty(targetPath) || !File.Exists(targetPath))
            {
                TxtStatus.Text = $"{styleName} Model Not Found.";
                BtnRun.IsEnabled = false;
                return;
            }

            await Task.Run(() => _engine.LoadModel(targetPath, useGpu));

            TxtStatus.Text = $"{styleName} Loaded ({_engine.DeviceMode})";
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

            ImgOutput.Source = null;
            BtnRun.IsEnabled = (_engine != null);
            BtnSave.IsEnabled = false;
            TxtStatus.Text = "Image Loaded.";
        }
    }

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_engine == null || _inputBytes == null) return;

        SetBusy(true, "Processing...");
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
        var dlg = new SaveFileDialog { Filter = "PNG|*.png", FileName = "anime_style.png" };
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
        CboStyle.IsEnabled = !busy; // 로딩 중 스타일 변경 방지
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _engine != null;
        BtnRun.IsEnabled = !busy && _engine != null && _inputBytes != null;
        if (msg != null) TxtStatus.Text = msg;
    }
}