#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# (C) 2021 Frederico Oliveira fred.santos.oliveira(at)gmail.com
#
#
import argparse
import sys
import os
from os import makedirs
from os.path import join, exists, basename, split
from glob import glob
from tqdm import tqdm
import librosa
import requests
import soundfile as sf
import json
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class Wav2Vec():
    def __init__(self, model_id='/root/katubev2/katube/w2v_ckpt12/'):

        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
    

    def transcribe(self, wav_file):
        audio, _ = librosa.load(wav_file)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        text = transcription[0]
        return text


class Whisper():

    def __init__(self, model_id='openai/whisper-large-v3'):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, 
                                                               torch_dtype=self.torch_dtype,
                                                                low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        
    def transcribe(self, wav_file):
        
        generate_kwargs= {"language":"<|pt|>","task": "transcribe", "num_beams": 5}
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            generate_kwargs=generate_kwargs,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        result = pipe(wav_file)
        text = result["text"]

        return text


class MMS:
    def __init__(self, model_id="facebook/mms-1b-all") -> None:
        self.model_id = model_id


            
    def transcribe(self, wav_file):
        pipe = pipeline(model=self.model_id, model_kwargs={"target_lang": "pt", "ignore_mismatched_sizes": True})
        result = pipe(wav_file)
        text = result["text"]

        return text
        




def convert_audios_samplerate(input_path, output_path, new_sample_rate):
    """
    Converts all audio files within a folder to a new sample rate.
        parameters:
            input_path: input folder path with wav files.
            output_path: output folder path to save converted wav files.

        Returns:
            Boolean: True of False.
    """

    if not(exists(output_path)):
        makedirs(output_path)

    for wavfile_path in tqdm(sorted(glob(input_path + "/*.wav"))):
        try:
            filename = basename(wavfile_path)
            data, sample_rate = librosa.load(wavfile_path)
            data = data.T
            new_data = librosa.resample(data, orig_sr=sample_rate, target_sr=new_sample_rate)
            output_file = join(output_path, filename)
            sf.write(output_file, new_data, new_sample_rate)
        except Exception as e:
            print('Error converting ' + wavfile_path)
            print(e) 
            return False

    return True


def get_transcript(wavefile_path):
    """
    Custom function to access a service STT. You must adapt it to use your contracted STT service.
        parameters:
            wavefile_path: wav filepath which will be transcribed.

        Returns:
            Text (str): Transcription of wav file.
    """
    with open(wavefile_path,'rb') as file_data:
        headers_raw = {
                'Content-Type': "application/x-www-form-urlencoded",
            	'endpointer.enabled': "true",
            	'endpointer.waitEnd': "5000",
            	'endpointer.levelThreshold': "5",
            	'decoder.confidenceThreshold': "10",
            	'decoder.maxSentences': "1",
            	'decoder.wordDetails': "0",
        }
        try:
            res = requests.post(url='https://your_url_here',
                                data=file_data,
                                headers=headers_raw)

            res.encoding='utf-8'
        except KeyboardInterrupt:
            print("KeyboardInterrupt Detected!")
            exit()
        except:
            #json_data=[{"message": "ERROR NO SPEECH"}]
            #return json_data
            return False
    return res.text

def transcribe_audios(input_path, output_file, model):
    """
    Iterate over the wav files inside a folder and transcribe them all.
        parameters:
            input_path: input wavs folder.
            output_file: output file to save the transcriptions following the template: "filename| transcription"

        Returns:
            Boolean: True or False.
    """

    out = open(output_file, 'w')

    for wavfile_path in tqdm(sorted(glob(input_path + "/*.wav")+ glob(input_path + "/*.mp3"))):
        filename = basename(wavfile_path)
        
        text = model.transcribe(wavfile_path)

        out.write("{}|{}\n".format(str(filename),str(text)))
    torch.cuda.empty_cache()
    out.close()
    return True




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--transcription_file', default='transcript.txt', help='Filename to save the transcripts')
    parser.add_argument('--input_dir', default='wavs', help='Directory of wav files')
    parser.add_argument('--temp_dir', default='wavs_16k', help='Directory to save wav files with sample rate (16k)')
    parser.add_argument('--new_sample_rate', default=16000,type=int, help='Sample rate used by the transcription api.')
    parser.add_argument('--whisper_model', default="openai/whisper-tiny", help='Whisper model used to transcription.')
    parser.add_argument('--mms_model', default="facebook/mms-1b-all", help='MMS model used to transcription.')
    parser.add_argument('--wav2vec_model', default="/root/katubev2/katube/w2v_ckpt12/", help='MMS model used to transcription.')
    

    args = parser.parse_args()

    input_path = join(args.base_dir, args.input_dir)
    converted_wavs_temp_path = join(args.base_dir,args.temp_dir)
    output_file = join(args.base_dir,args.transcription_file)

    Whisper_model = Whisper(args.whisper_model)

    # Convert audio sample rate
    print('Converting wav files...')
    convert_audios_samplerate(input_path, converted_wavs_temp_path, args.new_sample_rate)

    # Transcribe all wavs files
    print('Transcribing...')
    transcribe_audios(converted_wavs_temp_path, output_file, Whisper_model)


if __name__ == "__main__":
  main()
