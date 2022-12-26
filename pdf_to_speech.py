# importing required python modules
import argparse

# pytorch
import torch
import torchaudio
from tqdm import tqdm # for loop progress bar

# local methods
from file import PdfFile
from model import Model

def sort_arguments():
    pass

def main():

    # create parser object
    parser = argparse.ArgumentParser(description = "PDF text to speech")

    # add argument
    parser.add_argument("path", nargs = '*', metavar = "path", type = str,
            help = "ALL the pdf paths separted by spaces will be added.")

    # parse the arguments from standard input
    args = parser.parse_args()

    # check if add argument has any input data.
    if len(args.path) != 0:

        # loop through the arguments passed
        for arg in args.path:

            # process saving and loading for current file
            file = PdfFile(arg, audio_type='wav')

            # preprocess the pdf
            data = file.pre_process_pdf()

            model = Model(model_type=1, sample_rate=16000)

            # run the preprocessed data through the model
            waveform = model.text_to_speech(data=data, filename=file.filename, batch_size=15)

            # use pdf name as audio files name
            model.save_audio(waveform=waveform, path=file.audio_path)
    else:
        print('No file')

if __name__ == "__main__":
    main()

