#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# (C) 2021 Frederico Oliveira fred.santos.oliveira(at)gmail.com
#
#
from config import Config
from urllib.parse import parse_qs, urlparse
from search import search_videos
from transcribe import  transcribe_audios, Whisper
from utils.downsampling import downsampling
from validation import create_validation_file
from selection import select

import shutil
import os
import logging

from pyannote.audio import Model

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

# IMPORTANDO MODELOS!!!!
model_whisper = Whisper('openai/whisper-tiny')


# Argument Parser from File
'''
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            print(f.read().split())
            parser.parse_args(f.read().split(), namespace)
'''


def main():
    if Config.orig_base == 'channel':
        g = open(Config.channels_file, "r", encoding='utf-8')
    elif Config.orig_base == 'playlist':
        g = open(Config.playlists_file, "r", encoding='utf-8')
    else:
        g = None

    # Errors youtube videos file
    log_error_file = open(os.path.join(Config.logs_dir, Config.youtube_videos_error), "w")

    ######################################################
    # Youtube ignored videos
    ######################################################
    if Config.ignored_youtube_videos:
        try:
            f = open(Config.ignored_youtube_videos, encoding='utf-8')
            ignored_youtube_videos = f.readlines()
            f.close()
        except IOError:
          print("Error: File {} does not appear to exist.".format(Config.ignored_youtube_videos))
          return exit(False)

    ######################################################
    # Youtube already downloaded videos
    ######################################################
    if Config.downloaded_youtube_videos:
        try:
            if os.path.exists(Config.downloaded_youtube_videos):
                f = open(Config.downloaded_youtube_videos, "r", encoding='utf-8')
                downloaded_youtube_videos = f.readlines()
                f.close()
            else:
                f = open(Config.downloaded_youtube_videos, "w", encoding='utf-8')
                downloaded_youtube_videos = []
                f.close()                
        except IOError:
          print("Error: File {} does not appear to exist.".format(Config.downloaded_youtube_videos))
          return exit(False)

    ######################################################
    # Iterates over the youtube channels list
    ######################################################
    for content_id in g:
        content_id = content_id.rstrip()
        # ignore channel description
        if content_id.startswith('#'):
            print('Ignoring {}: {}'.format(Config.orig_base, content_id))
            continue
        # Defining output paths
        base_path = os.path.join(Config.base_dir, Config.dest_dir)
        output_path = os.path.join(base_path, Config.orig_base, content_id)
        print(base_path)
        ######################################################
        # Searching all videos from Youtube channel
        ######################################################
        print('Searching videos from {} - {}...'.format(Config.orig_base, content_id))
        # content_file contains the list of all videos on the youtube channel
        content_file = search_videos(Config.api_key, content_id, base_path, Config.output_search_file)
        if not content_file:
            logging.error('Error downloading channel video list: ' + content_id)
            continue

        # Open youtube videos list of the channel
        f = open(content_file, "r", encoding='utf-8')

        ######################################################
        # Iterate over youtube videos of the channel
        ######################################################
        i = 0
        for youtube_link in f:
            youtube_link = youtube_link.strip()
            ######################################################
            # Ignoring videos commented or found on list "Config.ignored_youtube_videos"
            ######################################################
            if youtube_link.startswith('#') or (Config.ignored_youtube_videos and youtube_link + '\n' in ignored_youtube_videos):
                print('Ignoring youtube video: {} '.format(youtube_link))
                continue

            videos = parse_qs(urlparse(youtube_link).query, keep_blank_values=True).get('v')
            video_id = None if videos == None else videos[0]

            wavs_dir = os.path.join(output_path, video_id)
            print('Converting {} - {}...'.format(i, youtube_link))
            tmp_wavs_dir = os.path.join(output_path, video_id, Config.tmp_wavs_dir)
            ######################################################
            # Transcribing: using Whisper/Wav2Vec/MMS
            ######################################################
            print('Transcribing {} - {}...'.format(i, youtube_link))
            transcription_file = os.path.join(output_path, video_id, Config.transcription_file)

            #por enquanto whisper
            if not transcribe_audios(wavs_dir, transcription_file, model_whisper):
                logging.error('YouTube video transcribing: ' + youtube_link)
                log_error_file.write(youtube_link + ': transcribe_audios' + '\n')
                # Removing temp dir
                shutil.rmtree(tmp_wavs_dir, ignore_errors=True)
                i += 1
                continue
            # Removing temp dir
            shutil.rmtree(tmp_wavs_dir, ignore_errors=True)
            
            comparision_file = os.path.join(output_path, video_id, Config.transcription_2_file)
            
            # OUTRO MODELO PODE SER INSERIDO AQUI
            if not transcribe_audios(wavs_dir, comparision_file, model_whisper):
                logging.error('YouTube video transcribing: ' + youtube_link)
                log_error_file.write(youtube_link + ': transcribe_audios' + '\n')
                # Removing temp dir
                shutil.rmtree(tmp_wavs_dir, ignore_errors=True)
                i += 1
                continue
            # Removing temp dir
            shutil.rmtree(tmp_wavs_dir, ignore_errors=True)

            ######################################################
            # Validating: using levenshtein distance
            ######################################################
            print('Validating {} - {}...'.format(i, youtube_link))
            basename = wavs_dir
            validation_file = os.path.join(output_path, video_id, Config.validation_file)
            if not create_validation_file(comparision_file, transcription_file, basename, validation_file):
                logging.error('YouTube video calculate distance: ' + youtube_link)
                log_error_file.write(youtube_link + ': create_validation_file'+ '\n')
                i += 1
                continue
            if Config.delete_temp_files:
                os.remove(comparision_file)
                os.remove(transcription_file)

            # ######################################################
            # # Selection: selecting only files with similarity (levenshtein) >= Config.minimal_levenshtein_distance
            # ######################################################
            print('Selection {} - {}...'.format(i, youtube_link))
            basename = wavs_dir
            output_filepath = os.path.join(output_path, video_id, Config.result_file)
            if not select(validation_file, output_filepath, Config.minimal_levenshtein_distance, Config.delete_temp_files):
                logging.error('YouTube video selection: ' + youtube_link)
                log_error_file.write(youtube_link + ': selection_file'+ '\n')
                i += 1
                continue
            if Config.delete_temp_files:
                os.remove(validation_file)

            ######################################################
            # Downsampling: downsampling wav files
            ######################################################            
            print('Downsampling {} - {}...'.format(i, youtube_link))
            if not downsampling(os.path.join(output_path, video_id), Config.wavs_dir, Config.tmp_wavs_dir, Config.sampling_rate, True):
                logging.error('YouTube video downsampling: ' + youtube_link)
                log_error_file.write(youtube_link + ': downsampling'+ '\n')
                i += 1
                continue

            if (os.path.exists(os.path.join(output_path, video_id, Config.tmp_wavs_dir))):
                os.rename(os.path.join(output_path, video_id, Config.tmp_wavs_dir), os.path.join(output_path, video_id, Config.wavs_dir))

            ######################################################
            # Excluding folders with no wav files
            ######################################################
            if not os.path.isdir(wavs_dir) or not os.listdir(wavs_dir):
                shutil.rmtree(os.path.join(output_path, video_id))

            print('Finish {} - {}...'.format(i, youtube_link))

            # Add youtube_link to already downloaded videos file
            if Config.downloaded_youtube_videos:
                with open(Config.downloaded_youtube_videos, 'a', encoding='utf-8') as out:            
                    out.write(youtube_link + "\n")    

            i += 1 # Next

        f.close() #  youtube videos list

    log_error_file.close()

    g.close() # channels or playlist list

if __name__ == "__main__":
    main()
