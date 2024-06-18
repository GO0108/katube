import argparse
# Dependências Necessárias:
#!pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
#!pip install deepfilternet==0.5.6
#!pip install silero-vad

import torch
import os
import tqdm

# Carregar o modelo Silero VAD
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model_vad)

def find_pauses(wav, start, end, vad_iterator, sampling_rate):
    pauses = []
    window_size_samples = int(sampling_rate * 0.15)  # 0.15 segundos em amostras
    for i in range(start, end, window_size_samples):
        chunk = wav[i:i + window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_dict = vad_iterator(chunk, return_seconds=False)
        if not speech_dict:
            pauses.append(i)
    vad_iterator.reset_states()  
    return pauses



def segment_audio(audio_path, output_dir, model_vad,vad_iterator, min_duration_sec=4, max_duration_sec=15, tgt_sampling_rate=48000):
    # Definição do caminho da pasta 'segmentos' e pasta de resultados
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    results_folder = os.path.join(output_dir, f"{base_name}")
    sampling_rate = 16000
    # Criação da pasta 'segmentos' se não existir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Criação da pasta de resultados dentro de 'segmentos' se não existir
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    wav_original = read_audio(audio_path, sampling_rate=sampling_rate)

    # Detectar segmentos de voz no áudio melhorado
    speech_timestamps = get_speech_timestamps(wav_original, model_vad, sampling_rate=sampling_rate)

    # Conversão de durações de segundos para amostras
    min_duration = min_duration_sec * sampling_rate  # min_duration em amostras
    max_duration = max_duration_sec * sampling_rate  # max_duration em amostras


    processed_segments = []
    for segment in tqdm.tqdm(speech_timestamps):
        start = segment['start']
        end = segment['end']
        duration = end - start

        if duration < min_duration:
            continue  # Descartar segmentos menores que min_duration
        elif duration > max_duration:
            # Encontrar pausas naturais dentro do segmento
            pauses = find_pauses(wav_original, start, end, vad_iterator, sampling_rate)
            if pauses:
                segment_start = start
                for pause in pauses:
                    if pause - segment_start >= min_duration and pause - segment_start <= max_duration:
                        processed_segments.append({'start': segment_start, 'end': pause})
                        segment_start = pause
                # Adicionar o último segmento
                if end - segment_start >= min_duration:
                    processed_segments.append({'start': segment_start, 'end': end})
            else:
                # Se não encontrar pausas, dividir normalmente
                for i in range(start, end, max_duration):
                    segment_end = min(i + max_duration, end)
                    processed_segments.append({'start': i, 'end': segment_end})
        else:
            processed_segments.append(segment)

    # Salvar segmentos processados do áudio original em arquivos separados
    os.makedirs(results_folder, exist_ok=True)
    wav_original = read_audio(audio_path, sampling_rate=tgt_sampling_rate)
    factor = int(tgt_sampling_rate/sampling_rate)

    for i, segment in enumerate(processed_segments):
        start = factor*segment['start']
        end = factor*segment['end']
        save_audio(os.path.join(results_folder, f"{base_name}_{i}.wav"), wav_original[start:end + factor*250], sampling_rate=tgt_sampling_rate)

    return True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--audio_file', default='audio.mp3', help='Filename to input audio file')
    parser.add_argument('--model_name', default='pyannote/segmentation', help='Model name for VAD')
    parser.add_argument('--output_dir', default='segments', help='Output dir')
    args = parser.parse_args()

    audio_path = os.path.join(args.base_dir, args.audio_file)
    model_name = args.model_name
    output_dir = os.path.join(args.base_dir, args.output_dir)
    
    model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    segment_audio(audio_path, output_dir, model)


if __name__ == "__main__":
    main()