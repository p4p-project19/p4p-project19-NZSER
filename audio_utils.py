from pydub import AudioSegment
from pydub.utils import make_chunks

### Utils to process audio files before feeding to the model ###

def get_audio_duration(audio_file):
    """
    Get the duration of an audio file in seconds.
    """
    audio_file = AudioSegment.from_file(audio_file)
    return audio_file.duration_seconds

def get_audio_chunks(signal, frame_size, sampling_rate):
    chunk_size = int(sampling_rate*frame_size*.001) # Chunk size = Sampling rate x frame size
    split_file = []
    for i in range(0, len(signal[0][0]), chunk_size):
        split_file.append(signal[0][0][i:chunk_size+i])
    return split_file

def export_audio_chunks(audio_file, chunk_length, output_dir):
    """
    Export the chunks of an audio file.
    """
    audio_file = AudioSegment.from_file(audio_file)
    chunks = make_chunks(audio_file, chunk_length)
    for i, chunk in enumerate(chunks):
        chunk.export(output_dir + "/chunk" + str(i) + ".wav", format="wav")
    return

