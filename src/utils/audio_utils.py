# Utils to process audio files before feeding to the model #
from pydub import AudioSegment
from pydub.utils import make_chunks


def get_audio_duration(audio_file):
    """
    Get the duration of an audio file in seconds.
    """
    audio_file = AudioSegment.from_file(audio_file)
    return audio_file.duration_seconds


def get_audio_chunks(signal, frame_size, sampling_rate, is_jl=False):
    """
    Returns a list of audio chunks from a signal. The chunks are of length
    specified by the frame_size parameter, and can be trimmed for matching annotations of
    the JL-Corpus.
    """
    # Chunk size = Sampling rate x frame size
    chunk_size = int(sampling_rate*frame_size*.001)
    split_file = []
    for i in range(0, len(signal), chunk_size):
        split_file.append(signal[i:chunk_size+i])
    if is_jl:
        split_file = split_file[1:-1]
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
