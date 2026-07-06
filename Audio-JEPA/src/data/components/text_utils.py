import subprocess
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize

class ArabicTextProcessor:
    def __init__(self):
        print("Initializing ArabicTextProcessor...")
        try:
            # We use the MLE disambiguator as a neural-like diacritizer fallback from CAMeL tools
            self.mle = MLEDisambiguator.pretrained('calima-msa-r13')
            print("CAMeL Tools MLEDisambiguator loaded successfully.")
        except Exception as e:
            print(f"Error loading CAMeL Tools MLEDisambiguator: {e}")
            self.mle = None

    def diacritize(self, text):
        if not self.mle:
            return text
            
        # 1. Normalize
        text = normalize_unicode(text)
        # 2. Tokenize
        tokens = simple_word_tokenize(text)
        # 3. Diacritize using MLE
        disambig = self.mle.disambiguate(tokens)
        diacritized_tokens = [d.analyses[0].diac for d in disambig]
        return ' '.join(diacritized_tokens)

    def phonemize(self, text):
        # phonemize with espeak-ng using IPA output
        try:
            result = subprocess.run(
                ["espeak-ng", "-v", "ar", "--ipa", "-q", text],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except FileNotFoundError:
            print("Error: espeak-ng is not installed or not in PATH.")
            return ""
        except subprocess.CalledProcessError as e:
            print(f"Error phonemizing text: {e}")
            return ""

    def process(self, text):
        diacritized = self.diacritize(text)
        phonemized = self.phonemize(diacritized)
        return {
            "original": text,
            "diacritized": diacritized,
            "phonemes": phonemized
        }

if __name__ == "__main__":
    processor = ArabicTextProcessor()
    sample_text = "مرحبا بك في مشروع تحويل النص إلى كلام"
    print(f"\nProcessing sample text: {sample_text}")
    output = processor.process(sample_text)
    print(f"Diacritized: {output['diacritized']}")
    print(f"Phonemes: {output['phonemes']}")
