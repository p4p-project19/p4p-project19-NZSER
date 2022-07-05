from pydub import AudioSegment
from pydub.utils import make_chunks

### Utils to process audio files before feeding to the model ###

def get_audio_duration(audio_file):
    """
    Get the duration of an audio file in seconds.
    """
    audio_file = AudioSegment.from_file(audio_file)
    return audio_file.duration_seconds

def get_audio_chunks_list(audio_file, chunk_length):
    """
    .
    
    Get the chunks of an audio file.
    """
    audio_file = AudioSegment.from_file(audio_file)
    return make_chunks(audio_file, chunk_length)

def export_audio_chunks(audio_file, chunk_length, output_dir):
    """
    Export the chunks of an audio file.
    """
    audio_file = AudioSegment.from_file(audio_file)
    chunks = make_chunks(audio_file, chunk_length)
    for i, chunk in enumerate(chunks):
        chunk.export(output_dir + "/chunk" + str(i) + ".wav", format="wav")
    return

