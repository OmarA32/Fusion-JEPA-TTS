
PADDING_TOKEN = '_pad_'
EOS_TOKEN = '_eos_'
DOUBLING_TOKEN = '_dbl_'
SEPARATOR_TOKEN = '_+_'

EOS_TOKENS = [SEPARATOR_TOKEN, EOS_TOKEN]

symbols = [
    # special tokens
    PADDING_TOKEN,  # padding
    EOS_TOKEN,  # eos-token
    '_sil_',  # silence
    DOUBLING_TOKEN,  # doubling
    SEPARATOR_TOKEN,  # word separator
    # consonants
    '<',  # hamza
    'b',  # baa'
    't',  # taa'
    '^',  # thaa'
    'j',  # jiim
    'H',  # Haa'
    'x',  # xaa'
    'd',  # daal
    '*',  # dhaal
    'r',  # raa'
    'z',  # zaay
    's',  # siin
    '$',  # shiin
    'S',  # Saad
    'D',  # Daad
    'T',  # Taa'
    'Z',  # Zhaa'
    'E',  # 3ayn
    'g',  # ghain
    'f',  # faa'
    'q',  # qaaf
    'k',  # kaaf
    'l',  # laam
    'm',  # miim
    'n',  # nuun
    'h',  # haa'
    'w',  # waaw
    'y',  # yaa'
    'v',  # /v/ for loanwords e.g. in u'fydyw': u'v i0 d y uu1',
    # vowels
    'a',  # short
    'u',
    'i',
    'aa',  # long
    'uu',
    'ii',
    # English ARPAbet Phonemes (g2p_en)
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 
    'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 
    'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 
    'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 
    'W', 'Y', 'Z', 'ZH'
]
