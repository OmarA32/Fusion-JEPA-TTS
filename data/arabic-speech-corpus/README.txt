###########################################################################################
# Package name: Arabic Speech Corpus                                                      #
# Author: Nawar HALABI (nawar.halabi@gmail.com)                                           #
# Owner: MicroLinkPC (UK) LIMITED (wpateam@microlinkpc.com)                               #
# Version: 2.0                                                                            #
# License: Commercial (Commercial use only allowed after agreement with OWNER).           #
#                                                                                         #
# Thank you for downloading the corpus. The corpus is free for research purposes. For a   #
# technical support or commercial license details, please contact OWNER or AUTHOR.        #
#                                                                                         #
# Please ignore the commerical license requirement if you have already aquired one        #
# through one of our distributors.                                                        #
###########################################################################################

1- the subdirectory 'test-set' contains an extra 100 utterances which were used to test
   speech made by the utterances in 'main-corpus'. It was recorded in a different time
   so it might be slightly different in acoustic nad phonetic features.

2- The 'lab' subdirectories contain label files with the orthographic transcirpt in
   them. Note that the file 'orthographic-transcript.txt' is simply all the 'lab'
   files merged and paired with their corresponding wav file name.

3- The 'textgrid' subdirectories contain praat files with the phonetic transcirpt  and
   timetamps (alignments). Please read more online about the textgrid format. Note that
   the file 'phonetic-transcript.txt' is simply all the phonetic transcripts merged
   without timestamps paired with their corresponding wav file name.

4- In orthographic transcripts a pause is resembled with a '-'. In phonetic transcripts
   it is resembled with 'sil'.
   
5- Any corrupt phonemes are marked with 'dist'. the `schwa` is marked as '-' in the
   phonetic transcript only (not included in the orthographic).

6- For more details on the phonemes used, please refer to my PhD thesis on the website.