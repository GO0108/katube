from config import Config
import argparse
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import os
from pydub import AudioSegment


def segment_audio(audio_path, output_dir, model):
    
    vad_model = VoiceActivityDetection(segmentation=model)
    vad_model.instantiate({
        "onset": 0.5, "offset": 0.5,
        "min_duration_on": 0.5,
        "min_duration_off": 0.2
    })

    # Definição do caminho da pasta 'segmentos' e pasta de resultados
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    results_folder = os.path.join(output_dir, f"{base_name}")

    # Criação da pasta 'segmentos' se não existir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Criação da pasta de resultados dentro de 'segmentos' se não existir
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # Processamento do arquivo de áudio
    vad = vad_model(audio_path)
    audio = AudioSegment.from_file(audio_path, 'mp3')

    for i, (segment, _, _) in enumerate(vad.itertracks(yield_label=True)):
        # Extraíndo o segmento de áudio baseado nos tempos de início e fim do segmento
        segment_audio = audio[segment.start*1000:segment.end*1000]
        # Exportando o segmento de áudio para um arquivo WAV no diretório especificado
        segment_audio.export(os.path.join(results_folder, f"{base_name}_{i}.wav"), format="wav")
    

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
    
    # Configuração do modelo de VAD carregado globalmente
    YOUR_HF_TOKEN = Config.HF_key
    model = Model.from_pretrained(model_name, use_auth_token=YOUR_HF_TOKEN)
    segment_audio(audio_path, output_dir, model)


if __name__ == "__main__":
    main()