# Dependences: 
# torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# deepfilternet==0.5.6
# Dependences: spleeter==2.4.0

from df.enhance import enhance, init_df, load_audio, save_audio
import os
# from spleeter.separator import Separator


# class SeparatorSingleton:
#     _separator = None

#     @staticmethod
#     def get_instance():
#         if SeparatorSingleton._separator is None:
#             SeparatorSingleton._separator = Separator('spleeter:2stems')
#         return SeparatorSingleton._separator

# Inicializa o modelo e o estado do DF assim que o módulo é importado
model, df_state, _ = init_df()

def denoise(input_path, model, df_state):
    try:
        audios = os.listdir(input_path)
        for audio_path in audios:
            audio_path = os.path.join(input_path, audio_path)
            if audio_path.lower().endswith(('.wav', '.mp3')):
                # Carrega o áudio
                audio, _ = load_audio(audio_path, sr=df_state.sr())

                # Realiza o processo de melhoramento do áudio
                enhanced_audio = enhance(model, df_state, audio)

                # Salva o áudio melhorado
                save_audio(audio_path, enhanced_audio, df_state.sr())

        return True

    except Exception as e:
        return f"Erro ao melhorar o áudio: {e}"



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
