class Config:
    base_dir = './'
    dest_dir = 'output'
    ## serch and ingest
    api_key = 'AIzaSyAUh28b6RfqDasvTqFLEl5mxwqCno_1jpA'
    HF_key = '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'
    ## videos origin
    orig_base = 'channel' # ['channel', 'playlist']
    ## Channels and Playlists files
    channels_file = './input/channels_id_example.txt'
    playlists_file = './input/playlists_id_example.txt'

    # Logs
    logs_dir = 'logs'
    youtube_videos_error = 'error_youtube_videos.txt'
    log_file = 'errors.log'
    # Ignore videos list
    ignored_youtube_videos = ''
    downloaded_youtube_videos = logs_dir + '/downloaded_youtube_videos.txt'

    output_search_file = 'youtube_videos.txt'
    # text_normalization
    min_words = 15
    max_words = 30
    # split_audio
    wavs_dir = 'wavs'
    metadata_subtitles_file = 'subtitles.csv'
    # convertion to transcribe format
    tmp_wavs_dir = 'wavs_tmp'
    tmp_sampling_rate = 24000
    # transcribe
    transcription_file = 'transcript.csv'
    transcription_2_file = 'transcript_2.csv'


    #output_converted_wavs_path = '00_24k'
    # validation
    validation_file = 'validation.csv'
    # selection
    minimal_levenshtein_distance = 0.9
    # downsampling
    sampling_rate = 24000
    # result
    result_file = 'metadata.csv'
    delete_temp_files = True