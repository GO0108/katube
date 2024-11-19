import os
import pandas as pd
import tqdm
from transcribe import Wav2Vec, Whisper

whisp = Whisper()

diretorio1 = '/app/data_2806'
diretorio2 = '/app/data_2808'

wavs = [os.path.join(root, file) 
        for diretorio in [diretorio1, diretorio2]
        for root, dirs, files in os.walk(diretorio) 
        for file in files 
        if file.endswith('.wav')]

output_file = 'transcriptions_whisper.csv'

downloaded = list(pd.read_csv(output_file).segment_name)
to_download = tqdm.tqdm([wav for wav in wavs if '/'.join(wav.split('/')[-3:]) not in downloaded])



for wav in tqdm.tqdm(to_download):
    segment_name = '/'.join(wav.split('/')[-3:])
    
    transcription = whisp.transcribe(wav)
    
    df = pd.read_csv(output_file)
    df.loc[len(df)] = {'segment_name': segment_name, 'transcription': transcription}
    df.to_csv(output_file, index=False)
