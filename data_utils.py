from datasets import load_dataset, load_metric, Audio
import os
import pandas as pd


def dataset_to_csv(dataset_dir: str, output_file: str):
    wavfile_data = []
    textfile_data = []
    for (root, dirs, files) in os.walk(dataset_dir, topdown=True):
        for fn in files:
            if fn.endswith(".wav"):
                wav_id = os.path.splitext(fn)[0]
                path = os.path.join(root, fn)
                wavfile_data.append((wav_id, fn, path))
            elif fn.endswith(".txt-utf8"):
                text_id = os.path.splitext(fn)[0]
                with open(os.path.join(root, fn)) as text_file:
                    text = text_file.read()
                textfile_data.append((text_id, text))
    df_wav = pd.DataFrame(wavfile_data, columns=["segment_id", "wav_file", "path"])
    df_wav = df_wav.set_index("segment_id")
    df_wav.to_csv(output_file)

def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def load_dataset_from_csv(data_files: list[str]):
    # load dataset from csv files
    dataset = load_dataset("csv", data_files=data_files)
    # split dataset
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.1)
    # loading audio
    dataset = dataset.cast_column("path", Audio())
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # preprocess dataset
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)
    return dataset

my_audio_dataset = load_dataset("data\msp_test", split="train")

print("SUP")

