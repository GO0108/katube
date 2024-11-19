# Dependences: 
# torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# deepfilternet==0.5.6
# Dependences: spleeter==2.4.0

from df.enhance import enhance, init_df, load_audio, save_audio
import os
# from spleeter.separator import Separator
from os import makedirs
from os.path import join, exists, basename, split
from glob import glob
from tqdm import tqdm
import librosa
import requests
import soundfile as sf
import json
import torch
import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance
"""
# Inicializa o modelo e o estado do DF assim que o módulo é importado
model, df_state, _ = init_df()

def denoise(input_path, model, df_state):
    try:
        sr = 24000
        audios = os.listdir(input_path)
        for audio_path in audios:
            audio_path = os.path.join(input_path, audio_path)
            if audio_path.lower().endswith(('.wav', '.mp3')):
                # Carrega o áudio
                audio, _ = load_audio(audio_path, sr=sr)

                # Realiza o processo de melhoramento do áudio
                enhanced_audio = enhance(model, df_state, audio)

                # Salva o áudio melhorado
                save_audio(audio_path, enhanced_audio, sr)

        return True

    except Exception as e:
        print('Error denoise ' + audio_path)
        print(e) 
        return False

"""

def denoise_audio(input_path):
    DEVICE =  'cuda'
    try:
        audios = os.listdir(input_path)
        for audio_path in audios:
            audio_path = os.path.join(input_path, audio_path)
            if audio_path.lower().endswith(('.wav')):
                # Carrega o áudio
                audio, sr = torchaudio.load(audio_path)
                audio = audio.mean(dim=0)
                # Realiza o processo de melhoramento do áudio
                wav, new_sr = denoise(audio, sr,DEVICE)
        
                # Salva o áudio melhorado
                sf.write(audio_path,  wav.numpy(), new_sr)
        return True

    except Exception as e:
        print('Error denoise ' + audio_path)
        print(e) 
        return False


def convert_audios_samplerate(input_path, new_sample_rate):
    """
    Converts all audio files within a folder to a new sample rate.
        parameters:
            input_path: input folder path with wav files.

        Returns:
            Boolean: True of False.
    """


    for audio_path in  tqdm(sorted(glob(input_path + "/*.wav"))):
        try:
            filename = basename(audio_path)
            data, sample_rate = librosa.load(audio_path)
            data = data.T
            new_data = librosa.resample(data, orig_sr=sample_rate, target_sr=new_sample_rate)
            sf.write(audio_path, new_data, new_sample_rate)
        except Exception as e:
            print('Error converting ' + audio_path)
            print(e) 
            return False

    return True

# separator = SeparatorSingleton.get_instance()  # Obtém a instância do Singleton
def separar_e_salvar_vocais(audio_path,separator):
    try:
        output_folder = "NoMusic"
        os.makedirs(output_folder, exist_ok=True)

        separator.separate_to_file(audio_path, output_folder)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        original_vocals_path = os.path.join(output_folder, base_name, "vocals.wav")
        new_vocals_path = os.path.join(output_folder, f"{base_name}_noMusic.wav")
        os.rename(original_vocals_path, new_vocals_path)

        accompaniment_path = os.path.join(output_folder, base_name, "accompaniment.wav")
        os.remove(accompaniment_path)
        os.rmdir(os.path.join(output_folder, base_name))

        return new_vocals_path

    except Exception as e:
        return f"Erro ao separar o áudio: {e}"
