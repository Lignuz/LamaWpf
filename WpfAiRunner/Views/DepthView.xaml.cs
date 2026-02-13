using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Microsoft.Win32;
using OnnxEngines.Depth;
using OnnxEngines.Utils;

namespace WpfAiRunner.Views;

public partial class DepthView : BaseAiView
{
    private DepthEstimator? _estimator;
    private string? _modelPath;
    private bool _hasInferenceResult = false;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    private Point? _lastClickPoint = null; // 마지막 클릭된 상대 좌표 저장

    public DepthView() => InitializeComponent();
    public override void Dispose() => _estimator?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        if (_estimator == null && string.IsNullOrEmpty(_modelPath))
        {
            string? debugPath = OnnxHelper.FindModelInDebug("depth_anything_v2_small.onnx");
            if (debugPath != null)
            {
                await ReloadModelAsync(debugPath);
            }
        }
#endif
        UpdateButtons();
    }

    protected override void OnImageLoaded()
    {
        ImgOutput.Source = null;
        _hasInferenceResult = false;
        UpdateButtons();
    }

    private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "ONNX (*.onnx)|*.onnx" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;
        await ReloadModelAsync(dlg.FileName);
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(_modelPath))
            await ReloadModelAsync(_modelPath);
    }

    private async Task ReloadModelAsync(string path)
    {
        SetBusyState(true);
        Log("Loading model...");
        await Task.Delay(10);

        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _estimator?.Dispose();

            _estimator = await Task.Run(() => new DepthEstimator(path, useGpu));
            _modelPath = path;
            _hasInferenceResult = false;
            ImgOutput.Source = null;

            TxtModel.Text = Path.GetFileName(path);
            Log($"Loaded on {_estimator.DeviceMode}");

            if (useGpu && _estimator.DeviceMode.Contains("CPU"))
            {
                ChkUseGpu.IsChecked = false;
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Load failed: {ex.Message}");
            Log("Load failed.");
        }
        finally
        {
            SetBusyState(false);
            UpdateButtons();
            GC.Collect();
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_estimator == null || _inputBitmap == null) return;

        SetBusyState(true);
        Log("Estimating...");

        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            await Task.Run(() => _estimator.RunInference(inputBytes));
            _hasInferenceResult = true;
            await UpdateResultImage();
            Log("Done.");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Failed.");
        }
        finally
        {
            SetBusyState(false);
            UpdateButtons();
            GC.Collect();
        }
    }

    private async void CboStyle_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_hasInferenceResult) return;
        SetBusyState(true);
        await UpdateResultImage();
        SetBusyState(false);
    }

    private async Task UpdateResultImage()
    {
        if (!_hasInferenceResult || _estimator == null) return;

        try
        {
            var style = (ColormapStyle)CboStyle.SelectedIndex;
            byte[] resultBytes = await Task.Run(() => _estimator.GetDepthMap(style));
            ImgOutput.Source = BytesToBitmap(resultBytes);
        }
        catch (Exception ex) { Log($"Style failed: {ex.Message}"); }
    }

    private void ImgInput_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (!_hasInferenceResult || _estimator == null || ImgInput.Source == null) return;

        Point p = e.GetPosition(ImgInput);

        // 실제 이미지 렌더링 영역 계산
        double actualWidth = ImgInput.ActualWidth;
        double actualHeight = ImgInput.ActualHeight;
        double sourceWidth = ImgInput.Source.Width;
        double sourceHeight = ImgInput.Source.Height;

        double ratio = Math.Min(actualWidth / sourceWidth, actualHeight / sourceHeight);
        double imgRenderWidth = sourceWidth * ratio;
        double imgRenderHeight = sourceHeight * ratio;

        double leftEdge = (actualWidth - imgRenderWidth) / 2;
        double topEdge = (actualHeight - imgRenderHeight) / 2;

        // 이미지 영역 내부 좌표 계산 (0.0 ~ 1.0)
        double relX = (p.X - leftEdge) / imgRenderWidth;
        double relY = (p.Y - topEdge) / imgRenderHeight;

        // 이미지 밖을 클릭한 경우 무시
        if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;

        // 좌표 저장 및 마커 이동
        _lastClickPoint = new Point(relX, relY);
        UpdateFocusMarkerPosition(p);

        ApplyRefocus(relX, relY);
    }

    // 노란색 마커 표시
    private void UpdateFocusMarkerPosition(Point p)
    {
        FocusMarker.Visibility = Visibility.Visible;
        Canvas.SetLeft(FocusMarker, p.X - (FocusMarker.Width / 2));
        Canvas.SetTop(FocusMarker, p.Y - (FocusMarker.Height / 2));
    }
    
    // 슬라이더 변경 시 호출되는 공통 핸들러
    private void SldBlur_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_hasInferenceResult) return;

        int mode = CboEffectMode.SelectedIndex;

        // Fog(2)와 Sky(3)는 클릭 좌표가 없어도 슬라이더만으로 실시간 업데이트 가능
        if (mode == 2 || mode == 3)
        {
            ApplyRefocus(0, 0);
        }
        // 그 외 모드는 클릭한 지점 정보가 있어야 업데이트
        else if (_lastClickPoint.HasValue)
        {
            ApplyRefocus(_lastClickPoint.Value.X, _lastClickPoint.Value.Y);
        }
    }

    // 창 크기가 바뀌어도 마커 위치를 유지하기 위한 로직
    private void ImgInput_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        // 1. 이미 분석 결과가 있고, 이전에 클릭한 좌표 정보가 있을 때만 실행
        if (!_hasInferenceResult || _lastClickPoint == null || ImgInput.Source == null)
        {
            if (FocusMarker != null) FocusMarker.Visibility = Visibility.Collapsed;
            return;
        }

        // 2. 바뀐 컨트롤 크기(ActualWidth/Height)를 기준으로 다시 여백과 렌더링 영역 계산
        double actualWidth = ImgInput.ActualWidth;
        double actualHeight = ImgInput.ActualHeight;
        double sourceWidth = ImgInput.Source.Width;
        double sourceHeight = ImgInput.Source.Height;

        double ratio = Math.Min(actualWidth / sourceWidth, actualHeight / sourceHeight);
        double imgRenderWidth = sourceWidth * ratio;
        double imgRenderHeight = sourceHeight * ratio;

        double leftEdge = (actualWidth - imgRenderWidth) / 2;
        double topEdge = (actualHeight - imgRenderHeight) / 2;

        // 3. 저장된 상대 좌표(_lastNormalizedPoint)를 이용해 현재의 절대 좌표 계산
        double newX = leftEdge + (_lastClickPoint.Value.X * imgRenderWidth);
        double newY = topEdge + (_lastClickPoint.Value.Y * imgRenderHeight);

        // 4. 마커 위치 업데이트
        Canvas.SetLeft(FocusMarker, newX - (FocusMarker.Width / 2));
        Canvas.SetTop(FocusMarker, newY - (FocusMarker.Height / 2));
        FocusMarker.Visibility = Visibility.Visible;
    }

    private async void ApplyRefocus(double relX, double relY)
    {
        if (_estimator == null || !_hasInferenceResult) return;

        try
        {
            float strength = (float)SldBlur.Value;
            byte[]? result = null;

            // 선택된 모드에 따라 엔진 메서드 호출
            switch (CboEffectMode.SelectedIndex)
            {
                case 0: // Blur (Focus)
                    result = await Task.Run(() => _estimator.RenderRefocus(relX, relY, strength));
                    break;
                case 1: // B&W (Isolation)
                    result = await Task.Run(() => _estimator.RenderColorIsolation(relX, relY, strength));
                    break;
                case 2: // Fog (Depth)
                    result = await Task.Run(() => _estimator.RenderFogEffect(strength));
                    break;
                case 3: // Sky (Replace)
                    if (_skyImageBytes == null) return;
                    // 슬라이더 0~30 범위를 하늘 인식 임계값(0.0~0.6)으로 변환
                    float threshold = strength / 50f;
                    result = await Task.Run(() => _estimator.RenderSkyReplacement(_skyImageBytes, threshold));
                    break;
                case 4: // Relight (조명 옵션 통합 반영)
                    var colorItem = CboLightColor.SelectedItem as ComboBoxItem;
                    string tagValue = colorItem?.Tag?.ToString() ?? "255,255,255";
                    var colorTag = tagValue.Split(',');
                    if (colorTag.Length == 3)
                    {
                        var lightColor = new OpenCvSharp.Vec3b(
                            byte.Parse(colorTag[0]),
                            byte.Parse(colorTag[1]),
                            byte.Parse(colorTag[2]));

                        result = await Task.Run(() => _estimator.RenderRelighting(relX, relY, strength, lightColor));
                    }
                    break;
            }

            if (result != null && result.Length > 0)
            {
                ImgOutput.Source = BytesToBitmap(result);
            }
        }
        catch (Exception ex) { Log($"Effect error: {ex.Message}"); }
    }

    // 리포커싱 취소 및 원래 결과(깊이 맵)로 복구
    private async void BtnResetFocus_Click(object sender, RoutedEventArgs e)
    {
        FocusMarker.Visibility = Visibility.Collapsed;
        await UpdateResultImage(); // 기존의 스타일 적용된 깊이 맵으로 복구
        Log("Focus reset to original depth map.");
    }

    private byte[]? _skyImageBytes = null;

    private void BtnLoadSky_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new Microsoft.Win32.OpenFileDialog { Filter = "Images|*.jpg;*.png;*.jpeg" };
        if (dialog.ShowDialog() == true)
        {
            _skyImageBytes = System.IO.File.ReadAllBytes(dialog.FileName);
            ApplyRefocus(0, 0); // 즉시 합성 실행
        }
    }

    private void CboEffectMode_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // 모드별 전용 UI 요소 가시성 제어
        if (BtnLoadSky != null)
            BtnLoadSky.Visibility = CboEffectMode.SelectedIndex == 3 ? Visibility.Visible : Visibility.Collapsed;

        if (PnlRelightOptions != null)
            PnlRelightOptions.Visibility = CboEffectMode.SelectedIndex == 4 ? Visibility.Visible : Visibility.Collapsed;

        // 모드 변경 즉시 화면 업데이트
        if (_hasInferenceResult)
        {
            if (CboEffectMode.SelectedIndex == 2 || CboEffectMode.SelectedIndex == 3)
                ApplyRefocus(0, 0);
            else if (_lastClickPoint.HasValue)
                ApplyRefocus(_lastClickPoint.Value.X, _lastClickPoint.Value.Y);
        }
    }

    private void UpdateButtons()
    {
        bool busy = ControlPbarLoading?.Visibility == Visibility.Visible;
        BtnPickModel.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy;
        BtnRun.IsEnabled = !busy && _estimator != null && _inputBitmap != null;
        ChkUseGpu.IsEnabled = !busy;
    }
}