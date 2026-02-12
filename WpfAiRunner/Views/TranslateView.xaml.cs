using System.Windows;
using System.Windows.Controls;
using GTranslate.Translators;

namespace WpfAiRunner.Views;

public partial class TranslateView : BaseAiView
{
    private readonly GoogleTranslator _translator = new();
    
    public override void Dispose() { _translator.Dispose(); }
    
    protected override Image? ControlImgInput => null;
    protected override Image? ControlImgOutput => null;
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
            string sourceText = TxtInput.Text;

            // 번역 대상 언어 코드 매핑: 한국어(ko), 영어(en)
            string targetLangCode = CboTargetLang.SelectedIndex == 0 ? "ko" : "en";

            // 비동기 번역 실행
            var result = await _translator.TranslateAsync(sourceText, targetLangCode);

            TxtOutput.Text = result.Translation;
            Log($"Translated to {result.TargetLanguage}");
        }
        catch (Exception ex)
        {
            Log($"Error: {ex.Message}");
            MessageBox.Show($"번역 중 오류가 발생했습니다: {ex.Message}");
        }
        finally
        {
            this.SetBusyState(false);
        }
    }

    private void BtnClear_Click(object sender, RoutedEventArgs e)
    {
        TxtInput.Clear();
        TxtOutput.Clear();
        Log("Cleared");
    }

    private void BtnCopy_Click(object sender, RoutedEventArgs e)
    {
        if (!string.IsNullOrEmpty(TxtOutput.Text))
        {
            Clipboard.SetText(TxtOutput.Text);
            Log("Copied to Clipboard");
        }
    }
}