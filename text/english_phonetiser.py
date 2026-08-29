import re
import inflect
import nltk

def _ensure_nltk_resources():
    resources = ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}') if res == 'punkt' else nltk.data.find(f'taggers/{res}') if 'tagger' in res else nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

_ensure_nltk_resources()

from g2p_en import G2p
_inflect = inflect.engine()
_g2p = G2p()

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

def _remove_commas(m):
    return m.group(1).replace(',', '')

def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')

def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'

def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))

def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')

def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

def normalize_text(text):
    text = normalize_numbers(text)
    # Replace hyphens, dashes, and underscores with spaces so compound words phonetize cleanly
    text = re.sub(r'[-–—_]', ' ', text)
    # basic lowercasing and cleaning
    text = text.lower()
    text = re.sub(r'[\'\"()\[\]{}]', '', text)
    # collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def english_to_phonemes(text):
    """
    Returns a list of ARPAbet phonemes and punctuation from g2p_en.
    Example: "Hello" -> ['HH', 'AH0', 'L', 'OW1']
    """
    normalized = normalize_text(text)
    phonemes = _g2p(normalized)
    return phonemes

def phonemes_to_tokens(phonemes, append_space=True):
    from text.symbols import SEPARATOR_TOKEN, EOS_TOKEN
    tokens = []
    for p in phonemes:
        if p == ' ':
            tokens.append(SEPARATOR_TOKEN)
        else:
            # g2p_en outputs like 'AH0', 'T', 'CH', etc.
            # We will use exactly what g2p_en outputs as the symbol.
            tokens.append(p)
            
    if append_space and len(tokens) > 0 and tokens[-1] != SEPARATOR_TOKEN:
        tokens.append(SEPARATOR_TOKEN)
        
    tokens.append(EOS_TOKEN)
    return tokens

def english_to_tokens(text, append_space=True):
    phonemes = english_to_phonemes(text)
    tokens = phonemes_to_tokens(phonemes, append_space=append_space)
    return tokens
