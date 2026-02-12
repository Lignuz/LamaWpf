using System.Windows;
using System.Windows.Controls;
using GTranslate.Translators;

namespace WpfAiRunner.Views;

public partial class TranslateView : BaseAiView
{
    private readonly GoogleTranslator _translator = new();
    
    public override void Dispose() { _translator.Dispose(); }

#pragma warning disable CS8764
    protected override Image? ControlImgInput => null;
    protected override Image? ControlImgOutput => null;
#pragma warning restore CS8764
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public TranslateView() => InitializeComponent();

    private async void BtnTranslate_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(TxtInput.Text)) return;

        this.SetBusyState(true);
        Log("Connecting to Translation Server...");

        try
        {
            // 언어 코드 매핑 로직
            string targetCode = GetLangCode(CboTargetLang);

            // 번역 실행
            var result = await _translator.TranslateAsync(TxtInput.Text, targetCode);

            TxtOutput.Text = result.Translation;
            Log($"Translated to {result.TargetLanguage}");
        }
        catch (Exception ex)
        {
            Log($"Error: {ex.Message}");
            MessageBox.Show($"번역 오류: {ex.Message}");
        }
        finally { this.SetBusyState(false); }
    }

    // 언어 스왑 기능: Source와 Target 언어를 서로 바꿉니다.
    private void BtnSwap_Click(object sender, RoutedEventArgs e)
    {
        // 'Auto Detect'는 Target이 될 수 없으므로 인덱스 보정이 필요함
        int sourceIdx = CboSourceLang.SelectedIndex;
        int targetIdx = CboTargetLang.SelectedIndex;

        // Source가 Auto Detect(0)인 경우를 제외하고 스왑 시도
        if (sourceIdx > 0)
        {
            CboSourceLang.SelectedIndex = targetIdx + 1; // Target 인덱스는 Source 기준 +1 밀려있음
            CboTargetLang.SelectedIndex = sourceIdx - 1;
        }
        Log("Languages Swapped");
    }

    private string GetLangCode(ComboBox combo)
    {
        // ComboBoxItem의 Content 텍스트를 기반으로 코드 반환
        string content = ((ComboBoxItem)combo.SelectedItem).Content.ToString()!;
        return content switch
        {
            "Korean" => "ko",
            "English" => "en",
            "Japanese" => "ja", 
            _ => "auto"
        };
    }

    private void BtnClear_Click(object sender, RoutedEventArgs e)
    {
        TxtInput.Clear();
        TxtOutput.Clear();
    }

    private void BtnCopy_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(TxtOutput.Text)) Clipboard.SetText(TxtOutput.Text);
    }

    // 결과를 입력창으로 가져오는 로직
    private void BtnMoveToInput_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(TxtOutput.Text)) return;

        // 1. 현재 결과창의 내용을 입력창으로 복사
        TxtInput.Text = TxtOutput.Text;

        // 2. 결과창은 비워줌 (다시 번역할 준비)
        TxtOutput.Clear();

        // 3. 언어 설정도 스왑해주는 것이 자연스러움 (옵션)
        // 만약 한->영 번역 후 결과를 가져왔다면 이제 영->한 번역을 할 가능성이 높기 때문
        BtnSwap_Click(null!, null!);

        Log("Result moved to input.");
    }
}