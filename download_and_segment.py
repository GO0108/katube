#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# (C) 2021 Frederico Oliveira fred.santos.oliveira(at)gmail.com
#
#
from config import Config
from urllib.parse import parse_qs, urlparse
from search import search_videos
from download import download_audio_and_subtitles_from_youtube
from text_normalization import create_normalized_text_from_subtitles_file
from segmentation import segment_audio
from transcribe import convert_audios_samplerate

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
model_vad = Model.from_pretrained('pyannote/segmentation', use_auth_token=Config.HF_key)


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

            ######################################################
            # Download mp3 from youtube_link
            ######################################################
            print('Downloading {} - {}...'.format(i, youtube_link))
            # Ignore videos with no portuguese caption or no caption at all
            if os.path.exists(os.path.join(output_path, video_id)) or (not download_audio_and_subtitles_from_youtube(youtube_link, output_path)):
                logging.error('YouTube video already downloaded or is unavailable: ' + youtube_link)
                log_error_file.write(youtube_link + ': ingest_dataset' + '\n')
                i += 1
                continue



            ######################################################
            # Normalizing text preparing to syncronizing text-audio
            ######################################################
            print('Normalizing text {} - {}...'.format(i, youtube_link))
            subtitle_file = os.path.join(output_path, video_id) + '/' + video_id + ".srt"
            text_file = os.path.join(output_path, video_id) + '/' + video_id + ".txt"
            if not create_normalized_text_from_subtitles_file(subtitle_file, text_file, Config.min_words, Config.max_words):
                logging.error('YouTube video creating normalized text from subtitles file: ' + youtube_link)
                log_error_file.write(youtube_link + ': create_normalized_text_from_subtitles_file' + '\n')
                i += 1
                continue
            if Config.delete_temp_files:
                os.remove(subtitle_file)


            ######################################################
            # Segmenting audio
            ######################################################
            audio_filename = video_id + ".mp3"
            audio_file = os.path.join(output_path, video_id, audio_filename)


            print('Segmenting audio {} - {}...'.format(i, youtube_link))
            if not segment_audio(audio_file, output_path , model_vad):
                logging.error('YouTube video segmenting audio: '  + youtube_link)
                log_error_file.write(youtube_link + ': segment_audio' + '\n')
                i += 1
                continue
            # Removing original audio file
            if Config.delete_temp_files:
                os.remove(audio_file)

            # ######################################################
            # # Converting audios: adjust audios to transcription tool
            # ######################################################
                
            # wavs_dir = os.path.join(output_path, video_id, Config.wavs_dir)
            wavs_dir = os.path.join(output_path, video_id)
            print('Converting {} - {}...'.format(i, youtube_link))
            tmp_wavs_dir = os.path.join(output_path, video_id, Config.tmp_wavs_dir)
            if not convert_audios_samplerate(wavs_dir, tmp_wavs_dir, Config.tmp_sampling_rate):
                logging.error('YouTube video converting audio: ' + youtube_link)
                log_error_file.write(youtube_link  + ': convert_audios_samplerate' + '\n')
                i += 1
                continue

            i += 1 # Next

        f.close() #  youtube videos list

    log_error_file.close()

    g.close() # channels or playlist list

if __name__ == "__main__":
    main()
