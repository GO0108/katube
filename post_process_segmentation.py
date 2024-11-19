    #!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# (C) 2021 Frederico Oliveira fred.santos.oliveira(at)gmail.com
#
#
from config import Config
from urllib.parse import parse_qs, urlparse
from search import search_videos

from pydub import AudioSegment
from download import download_audio_and_subtitles_from_youtube
from text_normalization import create_normalized_text_from_subtitles_file
from segmentation import segment_audio
from transcribe import  transcribe_audios, Whisper, Wav2Vec
from utils.downsampling import downsampling
from validation import create_validation_file
from selection import select
from enhance import convert_audios_samplerate,denoise_audio

import torch
import shutil
import os
import logging

from pyannote.audio import Model
from df.enhance import init_df


import torch
import shutil
import os
import logging
import glob
import shutil
import mutagen
import os

path = '/app/katube-gama/output/channel/'
# tirando outlier
file_paths = []
for folder in os.listdir(path):
    for video in os.listdir(os.path.join(path, folder)):
        for wav_path in glob.glob(os.path.join(path, folder, video) + "/*.wav"):
            if mutagen.File(wav_path).info.length > 20:
                print(os.path.join(path, folder, video, wav_path))
                file_paths.append(os.path.join(path, folder, video, wav_path))
                #shutil.rmtree(os.path.join(path, folder, video))


# Salvando a lista de caminhos em um arquivo .txt
with open('/app/katube-gama/file_paths_postprocessed.txt', 'w') as f:
    for file_path in file_paths:
        f.write(f"{file_path}\n")



######################################################
# Logs Config
######################################################
if not(os.path.exists(Config.logs_dir)):
    os.makedirs(Config.logs_dir)

log_path = os.path.join(Config.logs_dir, Config.log_file)
if not os.path.exists(Config.logs_dir):
    os.makedirs(Config.logs_dir)
open(log_path, 'w').close()

level = logging.DEBUG # Options: logging.DEBUG | logging.INFO | logging.WARNING | logging.ERROR | logging.CRITICAL
logging.basicConfig(filename=log_path, filemode='w', format='%(message)s', level=level)


# Carregar o modelo Silero VAD
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:v4.0', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model_vad)

#model, df_state, _ = init_df()

# Argument Parser from File
'''
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            print(f.read().split())
            parser.parse_args(f.read().split(), namespace)
'''


def main():
    i=0
    for audio_file in file_paths:

        youtube_link = audio_file
        wavs_dir = "/".join(audio_file.split("/")[:-1])
        output_path =  "/".join(audio_file.split("/")[:-2])
        video_id = audio_file.split("/")[-2]
        

        ######################################################
        # Enhancing: enhance quality of audio
        ######################################################  
        print('Denoising {} - {}...'.format(i, youtube_link))
        if not denoise_audio(wavs_dir):
            logging.error('YouTube video denoise: ' + youtube_link)

            i += 1
            continue


        ######################################################
        # Segmenting audio
        ######################################################

        print('Segmenting audio {} - {}...'.format(i, youtube_link))
        if not segment_audio(audio_file, output_path , model_vad, vad_iterator):
            logging.error('YouTube video segmenting audio: '  + youtube_link)
            i += 1
            continue
        # Removing original audio file
        if Config.delete_temp_files:
            os.remove(audio_file)

        ######################################################
        # Converting audios: adjust audios to transcription tool
        ######################################################

        print('Converting {} - {}...'.format(i, youtube_link))
        tmp_wavs_dir = os.path.join(output_path, video_id, Config.tmp_wavs_dir)
        if not convert_audios_samplerate(wavs_dir,  Config.tmp_sampling_rate):
            logging.error('YouTube video converting audio: ' + youtube_link)
            i += 1
            continue
        
        ######################################################
        # Excluding folders with no wav files
        ######################################################
        if not os.path.isdir(wavs_dir) or not os.listdir(wavs_dir):
            shutil.rmtree(os.path.join(output_path, video_id))

        print('Finish {} - {}...'.format(i, youtube_link))

     
        # Removing temp dir
        shutil.rmtree(tmp_wavs_dir, ignore_errors=True)
        i+=1


    



if __name__ == "__main__":
    main()
