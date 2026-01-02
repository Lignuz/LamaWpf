using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using OnnxEngines.Sam;
using OnnxEngines.Utils;
using SamEngine;
using Path = System.IO.Path;

namespace WpfAiRunner.Views;

public partial class SamView : UserControl, IDisposable
{
    private ISamSegmenter _segmenter;
    private BitmapSource? _inputBitmap;
    private bool _isModelLoaded;
    private bool _isImageEncoded;
    private Point? _lastClickRatio;
    private bool _isUpdatingCombo;

    private string? _currentEncoderPath;
    private string? _currentDecoderPath;

    public SamView()
    {
        InitializeComponent();
        _segmenter = new Sam2Segmenter(); // 기본값 SAM 2
    }

    public void Dispose() => _segmenter?.Dispose();

    // 1. 화면 로드 시: 현재 선택된 모드(SAM 2)에 맞춰 자동 로드 시도
    private async void UserControl_Loaded(object sender, RoutedEventArgs e)
    {
        await AutoLoadModelForCurrentType();
    }

    // 2. 콤보박스 변경 시: 해당 모드로 엔진 교체 후 자동 로드 시도
    private async void CboModelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // 아직 UI 초기화 중이라면 스킵
        if (!IsLoaded) return;

        await AutoLoadModelForCurrentType();
    }

    // 현재 선택된 모드에 따라 엔진을 교체하고, Debug 모델을 찾아 로드하는 공통 함수
    private async Task AutoLoadModelForCurrentType()
    {
        int index = CboModelType.SelectedIndex;

        // 1. 엔진 인스턴스 교체
        _segmenter.Dispose();
        if (index == 0) _segmenter = new SamSegmenter();       // MobileSAM
        else _segmenter = new Sam2Segmenter();      // SAM 2

        // UI 상태 초기화 (모델 바뀜 -> 기존 이미지 인코딩 무효화)
        _isModelLoaded = false;
        _isImageEncoded = false;
        TxtStatus.Text = "Model changed. Please load weights.";

        // 2. 디버그 모드라면 자동 파일 찾기 시도
#if DEBUG
        string? encoder = null;
        string? decoder = null;

        if (index == 0) // MobileSAM
        {
            encoder = OnnxHelper.FindModelInDebug("mobile_sam.encoder.onnx");
            decoder = OnnxHelper.FindModelInDebug("mobile_sam.decoder.onnx");
        }
        else // SAM 2
        {
            encoder = OnnxHelper.FindModelInDebug("sam2_hiera_small.encoder.onnx");
            decoder = OnnxHelper.FindModelInDebug("sam2_hiera_small.decoder.onnx");
        }

        // 파일이 둘 다 있으면 바로 로딩
        if (encoder != null && decoder != null)
        {
            await LoadModelsInternal(encoder, decoder);
        }
#endif
    }

    private async void BtnLoadModels_Click(object sender, RoutedEventArgs e)
    {
        int modelTypeIndex = CboModelType.SelectedIndex;	// 0: MobileSAM, 1: SAM 2

        // 혹시 모르니 엔진 한번 더 확실히 리셋
        _segmenter.Dispose();
        if (modelTypeIndex == 0) _segmenter = new SamSegmenter();
        else _segmenter = new Sam2Segmenter();

        var dlg = new OpenFileDialog { Title = $"Select {CboModelType.Text} Encoder", Filter = "ONNX|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        string encoderPath = dlg.FileName;
        string folder = Path.GetDirectoryName(encoderPath)!;
        string encoderNameLower = Path.GetFileName(encoderPath).ToLower();

        string? decoderPath = null;
        var allDecoders = Directory.GetFiles(folder, "*decoder*.onnx");

        if (modelTypeIndex == 1) // SAM 2
        {
            string[] variants = { "tiny", "small", "base_plus", "large" };
            string? detectedVariant = variants.FirstOrDefault(v => encoderNameLower.Contains(v));

            if (!string.IsNullOrEmpty(detectedVariant))
            {
                decoderPath = allDecoders.FirstOrDefault(f =>
                    Path.GetFileName(f).ToLower().Contains(detectedVariant) &&
                    Path.GetFileName(f).ToLower().Contains("sam2"));
            }
            decoderPath ??= allDecoders.FirstOrDefault(f =>
            {
                string name = Path.GetFileName(f).ToLower();
                return name.Contains("sam2") || name.Contains("hiera");
            });
        }
        else // MobileSAM
        {
            decoderPath = allDecoders.FirstOrDefault(f => Path.GetFileName(f).ToLower().Contains("mobile"));
            decoderPath ??= allDecoders.FirstOrDefault(f =>
            {
                string name = Path.GetFileName(f).ToLower();
                return !name.Contains("sam2") && !name.Contains("hiera");
            });
        }

        decoderPath ??= encoderPath;

        if (decoderPath == encoderPath ||
            MessageBox.Show($"Use decoder: {Path.GetFileName(decoderPath)}?", "Confirm", MessageBoxButton.YesNo) == MessageBoxResult.No)
        {
            var dlg2 = new OpenFileDialog { Title = $"Select {CboModelType.Text} Decoder", Filter = "ONNX|*.onnx", InitialDirectory = folder };
            if (dlg2.ShowDialog(Window.GetWindow(this)) != true) return;
            decoderPath = dlg2.FileName;
        }

        await LoadModelsInternal(encoderPath, decoderPath);
    }

    // GPU 체크박스 클릭 시 모델 재로딩
    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        // 로드된 모델이 없으면 무시
        if (!_isModelLoaded || string.IsNullOrEmpty(_currentEncoderPath) || string.IsNullOrEmpty(_currentDecoderPath))
            return;

        // 현재 기억된 경로로 다시 로딩
        await LoadModelsInternal(_currentEncoderPath, _currentDecoderPath);
    }

    // 모델 로딩 내부 로직 (중복 제거 및 경로 저장)
    private async Task LoadModelsInternal(string encoderPath, string decoderPath)
    {
        SetBusy(true, "Loading models...");
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            await Task.Run(() => _segmenter.LoadModels(encoderPath, decoderPath, useGpu));

            // 성공 시 경로 저장
            _currentEncoderPath = encoderPath;
            _currentDecoderPath = decoderPath;
            _isModelLoaded = true;

            TxtStatus.Text = $"{((ComboBoxItem)CboModelType.SelectedItem).Content} Loaded ({_segmenter.DeviceMode})";
            BtnOpenImage.IsEnabled = true;

            // GPU 실패 시 UI 동기화 (Fallback 알림)
            if (useGpu && _segmenter.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
                MessageBox.Show("GPU init failed. Fallback to CPU.", "Info");
            }

            // 이미지가 있으면 재인코딩
            if (_inputBitmap != null)
            {
                await EncodeCurrentInput();
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading models:\n{ex.Message}");
            _isModelLoaded = false;
            _currentEncoderPath = null;
            _currentDecoderPath = null;
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void BtnOpenImage_Click(object sender, RoutedEventArgs e)
    {
        if (!_isModelLoaded) return;

        var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

        // 1. 이미지 로드
        var bmp = new BitmapImage();
        bmp.BeginInit();
        bmp.UriSource = new Uri(dlg.FileName);
        bmp.CacheOption = BitmapCacheOption.OnLoad;
        bmp.EndInit();
        bmp.Freeze();

        // 2. DPI 정규화 (좌표 계산 및 마스크 정합성 확보를 위해 필수)
        _inputBitmap = NormalizeDpi96(bmp);
        ImgInput.Source = _inputBitmap;
        ImgMask.Source = null;
        PointOverlay.Children.Clear();
        _lastClickRatio = null;

        // UI 초기화
        CboMaskCandidates.Items.Clear();
        CboMaskCandidates.IsEnabled = false;
        TxtOverlay.Visibility = Visibility.Visible;

        // 인코딩 실행
        await EncodeCurrentInput();
    }

    // 현재 입력 이미지 인코딩 (GPU 변경 시 재사용)
    private async Task EncodeCurrentInput()
    {
        if (_inputBitmap == null || !_isModelLoaded) return;

        SetBusy(true, "Encoding...");
        try
        {
            _isImageEncoded = false;
            byte[] bytes = BitmapToPngBytes(_inputBitmap);
            await Task.Run(() => _segmenter.EncodeImage(bytes));

            _isImageEncoded = true;
            TxtStatus.Text = "Ready. Click image.";
            TxtOverlay.Visibility = Visibility.Collapsed;

            BtnReset.IsEnabled = true;
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Encoding failed: {ex.Message}");
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void ImgInput_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (!_isImageEncoded || _inputBitmap == null) return;

        // 컨트롤 내 마우스 좌표
        Point viewPoint = e.GetPosition(ImgInput);

        // 실제 이미지가 렌더링된 영역 계산 (Letterbox 제외)
        Rect renderRect = GetImageRenderRect(ImgInput, _inputBitmap);
        if (!renderRect.Contains(viewPoint)) return;

        // 클릭 위치의 상대 비율(0.0 ~ 1.0) 계산
        double ratioX = (viewPoint.X - renderRect.X) / renderRect.Width;
        double ratioY = (viewPoint.Y - renderRect.Y) / renderRect.Height;
        _lastClickRatio = new Point(ratioX, ratioY);

        UpdateOverlayPoint();

        // 엔진에는 원본 이미지의 픽셀 좌표를 전달 (엔진 내부에서 Scale 계산)
        float targetX = (float)(ratioX * _inputBitmap.PixelWidth);
        float targetY = (float)(ratioY * _inputBitmap.PixelHeight);

        TxtStatus.Text = $"Segmenting {Math.Round(targetX)},{Math.Round(targetY)}...";
        try
        {
            // 인터페이스를 통한 예측 (MobileSAM 또는 SAM 2)
            var result = await Task.Run(() => _segmenter.Predict(targetX, targetY));

            if (result.Scores.Count > 0)
            {
                // 1. 1등 마스크 즉시 표시
                if (result.BestMaskBytes.Length > 0)
                {
                    var maskBmp = BytesToBitmap(result.BestMaskBytes);
                    ImgMask.Source = NormalizeDpi96(maskBmp);
                }

                // 2. 콤보박스 업데이트
                _isUpdatingCombo = true;
                CboMaskCandidates.SelectionChanged -= CboMaskCandidates_SelectionChanged;
                CboMaskCandidates.Items.Clear();
                CboMaskCandidates.IsEnabled = true;

                // 점수 내림차순 정렬된 인덱스 리스트 생성
                var sortedIndices = result.Scores
                                          .Select((s, i) => new { Score = s, Index = i })
                                          .OrderByDescending(x => x.Score)
                                          .ToList();

                foreach (var item in sortedIndices)
                {
                    var cbi = new ComboBoxItem
                    {
                        Content = $"Mask {item.Index + 1} ({item.Score:P1})",
                        Tag = item.Index
                    };
                    CboMaskCandidates.Items.Add(cbi);
                }

                // BestIndex 선택
                for (int i = 0; i < CboMaskCandidates.Items.Count; i++)
                {
                    if (CboMaskCandidates.Items[i] is ComboBoxItem item &&
                        (int)item.Tag == result.BestIndex)
                    {
                        CboMaskCandidates.SelectedIndex = i;
                        break;
                    }
                }

                // BestIndex가 없을 경우(거의 없음) 0번 선택
                if (CboMaskCandidates.SelectedIndex < 0)
                    CboMaskCandidates.SelectedIndex = 0;

                CboMaskCandidates.SelectionChanged += CboMaskCandidates_SelectionChanged;
                _isUpdatingCombo = false;

                TxtStatus.Text = "Done.";
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.ToString());
        }
    }

    private async void CboMaskCandidates_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // 콤보박스 자동 세팅 중일 땐 이미지 생성 스킵
        if (_isUpdatingCombo) return;
        if (CboMaskCandidates.SelectedItem is not ComboBoxItem selectedItem) return;
        if (selectedItem.Tag is not int maskIndex) return;

        TxtStatus.Text = "Rendering Mask...";

        byte[] maskBytes = await Task.Run(() => _segmenter.GetMaskImage(maskIndex));

        if (maskBytes.Length > 0)
        {
            var maskBmp = BytesToBitmap(maskBytes);
            ImgMask.Source = NormalizeDpi96(maskBmp);
            TxtStatus.Text = $"Selected: {selectedItem.Content}";
        }
    }

    private void ImgInput_SizeChanged(object sender, SizeChangedEventArgs e) => UpdateOverlayPoint();

    /// <summary>
    /// 클릭한 지점에 빨간 점을 표시합니다. 
    /// </summary>
    private void UpdateOverlayPoint()
    {
        PointOverlay.Children.Clear();
        if (_lastClickRatio == null || _inputBitmap == null) return;

        Rect renderRect = GetImageRenderRect(ImgInput, _inputBitmap);

        // 비율 좌표를 현재 컨트롤 크기에 맞는 좌표로 변환
        double drawX = (_lastClickRatio.Value.X * renderRect.Width) + renderRect.X;
        double drawY = (_lastClickRatio.Value.Y * renderRect.Height) + renderRect.Y;

        Point p = ImgInput.TranslatePoint(new Point(drawX, drawY), PointOverlay);

        var ell = new Ellipse
        {
            Width = 10,
            Height = 10,
            Fill = Brushes.Red,
            Stroke = Brushes.White,
            StrokeThickness = 2
        };
        Canvas.SetLeft(ell, p.X - 5);
        Canvas.SetTop(ell, p.Y - 5);
        PointOverlay.Children.Add(ell);
    }

    /// <summary>
    /// Image 컨트롤 내부에서 실제 이미지가 그려지는 영역(Rect)을 계산합니다.
    /// (Uniform Stretch 모드에서 발생하는 검은 여백을 제외한 영역)
    /// </summary>
    private Rect GetImageRenderRect(Image imgControl, BitmapSource bmp)
    {
        double ctrlW = imgControl.ActualWidth;
        double ctrlH = imgControl.ActualHeight;
        double bmpW = bmp.Width;
        double bmpH = bmp.Height;

        if (ctrlW <= 0 || ctrlH <= 0 || bmpW <= 0 || bmpH <= 0) return Rect.Empty;

        double aspectControl = ctrlW / ctrlH;
        double aspectImage = bmpW / bmpH;

        double renderW, renderH;

        if (aspectControl > aspectImage) // 컨트롤이 더 넓음 (좌우 여백)
        {
            renderH = ctrlH;
            renderW = ctrlH * aspectImage;
        }
        else // 컨트롤이 더 높음 (상하 여백)
        {
            renderW = ctrlW;
            renderH = ctrlW / aspectImage;
        }

        double offsetX = (ctrlW - renderW) / 2.0;
        double offsetY = (ctrlH - renderH) / 2.0;

        return new Rect(offsetX, offsetY, renderW, renderH);
    }

    private void BtnReset_Click(object sender, RoutedEventArgs e)
    {
        ImgMask.Source = null;
        PointOverlay.Children.Clear();
        _lastClickRatio = null;

        CboMaskCandidates.Items.Clear();
        CboMaskCandidates.IsEnabled = false;

        TxtStatus.Text = "Mask cleared.";
    }

    private void SetBusy(bool busy, string? msg = null)
    {
        PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        BtnLoadModels.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _isModelLoaded;
        BtnReset.IsEnabled = !busy && _isModelLoaded;
        ImgInput.IsEnabled = !busy;

        if (msg != null) TxtStatus.Text = msg;
    }

    private static byte[] BitmapToPngBytes(BitmapSource bmp)
    {
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        using var ms = new MemoryStream();
        enc.Save(ms);
        return ms.ToArray();
    }

    private static BitmapImage BytesToBitmap(byte[] bytes)
    {
        var img = new BitmapImage();
        using var ms = new MemoryStream(bytes);
        img.BeginInit();
        img.CacheOption = BitmapCacheOption.OnLoad;
        img.StreamSource = ms;
        img.EndInit();
        img.Freeze();
        return img;
    }

    /// <summary>
    /// 다양한 DPI를 가진 이미지를 96 DPI로 통일하여 좌표 계산 오차를 방지합니다.
    /// </summary>
    private static BitmapSource NormalizeDpi96(BitmapSource src)
    {
        if (src == null) throw new ArgumentNullException(nameof(src));
        const double dpi = 96.0;

        if (Math.Abs(src.DpiX - dpi) < 0.01 && Math.Abs(src.DpiY - dpi) < 0.01)
            return src;

        int w = src.PixelWidth;
        int h = src.PixelHeight;
        var pf = src.Format;

        if (pf == PixelFormats.Indexed1 || pf == PixelFormats.Indexed2 ||
            pf == PixelFormats.Indexed4 || pf == PixelFormats.Indexed8)
        {
            src = new FormatConvertedBitmap(src, PixelFormats.Bgra32, null, 0);
            pf = PixelFormats.Bgra32;
        }

        int bpp = (pf.BitsPerPixel + 7) / 8;
        int stride = w * bpp;
        byte[] buf = new byte[h * stride];
        src.CopyPixels(buf, stride, 0);

        var normalized = BitmapSource.Create(w, h, dpi, dpi, pf, null, buf, stride);
        normalized.Freeze();
        return normalized;
    }
}