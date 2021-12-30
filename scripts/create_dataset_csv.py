import os
import glob
import pandas as pd
import wave
import contextlib
from pathlib import Path


if __name__ == "__main__":
    audio_files = []
    for filename in glob.glob(
        str(Path(__file__).parent.parent / ".data/genres/**/*.wav")
    ):
        with open(os.path.join(os.getcwd(), filename), "r") as f, contextlib.closing(
            wave.open(filename, "r")
        ) as wf:

            file_name = os.path.basename(f.name)
            file_path = os.path.realpath(f.name)

            genre = file_name.split(".")[0]

            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

            file_entry = {
                "name": file_name,
                "path": file_path,
                "genre": genre,
                "frames": frames,
                "frame_rate": rate,
                "duration": duration,
            }

            audio_files.append(file_entry)

    audio_df = pd.DataFrame(audio_files)

    audio_df.to_csv(Path(__file__).parent.parent / ".data/data.csv")
