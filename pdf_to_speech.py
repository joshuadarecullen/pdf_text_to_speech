# importing required python modules
import os
from itertools import zip_longest
import argparse
import re

from num2words import num2words
from tqdm import tqdm # for loop progress bar
from PyPDF2 import PdfReader # pdf reader module
from nltk.tokenize import sent_tokenize # tokeniser, unused atm
from nltk.tokenize import word_tokenize as word_tokeniser# tokeniser, unused atm

import torch
import torchaudio


class File:
    def __init__(self, path: str, audio_type:str) -> None:
        self.INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .txt file."
        self.INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."
        
        self.validate_file(path=path)
        self.process_paths(audio_type)     

    def process_paths(self, audio_type: str) -> None:

        split_path = self.path.split('/') #  split at '/'
        self.filename = split_path[-1] # collect filename
        reduce_filename = re.sub('\.', '-', str(self.filename)) # remove '.'

        # craft audio path
        audio_path = '/'.join(split_path[1:-1])
        #for toke in split_path[1:-1]:
            #audio_path += '/'+toke

        self.audio_path = audio_path + f"/{reduce_filename}.{audio_type}"


    # validate the filen name
    def validate_file(self, path: str) -> None:
        '''
        validate file name and path.
        '''
        print(f'\nValidating {path}')

        if not self.valid_path(path=path):
            print(self.INVALID_PATH_MSG%(path))
            quit()
        elif not self.validate_filetype(path=path):
            print(self.INVALID_FILETYPE_MSG%(path))
            quit()

        print('Validation Successful...')
        self.path = path

    # check the file is a pdf
    def validate_filetype(self, path: str) -> bool:
        # validate file type
        return path.endswith('.pdf')

    # check the file exists
    def valid_path(self, path: str) -> bool:
        # validate file path
        return os.path.exists(path)

    # validate file then read the text
    def read(self) -> PdfReader:
        pass


class PdfFile(File):
    def __init__(self, path: str, audio_type: str) -> None:
        super().__init__(path=path, audio_type=audio_type)
        self.read()

    def read(self) -> None:
        # read and print the file content
        # pdfFileObj = open(file_path, 'rb')
        print('\nReading PDF...')
        self.reader = PdfReader(self.path)

    # pre process the data collected from the pdf
    def pre_process_pdf(self) -> str:
        all_pages = [] # collect all the text

        print('\nProcessing the text from each page of the pdf')
        for i in tqdm(range(self.reader.numPages)):
            page = self.reader.pages[i]
            text = self.process_text(page.extractText())

            all_pages += text if type(text)==list else all_pages.append(text)
            # sents = sent_tokenize(text)

        return all_pages

    def process_text(self, sent: str) -> str:
        proc_text = re.sub('\n', ' ', sent) # remove new line breaks
        tokenlist = proc_text.split() # split the text at every space
        # proc_text = [token for token in word_tokeniser(sent) if token.isalpha()]
        tokenlist = [re.sub(num2words(token, to = 'ordinal_num'), ' ') if (token.endswith(("nd","st","th")) and token[:-2].isdigit()) else token for token in tokenlist]
        tokenlist = [re.sub(num2words(str(token)[:4], to = 'ordinal'), ' ') if token.isdigit() else token for token in tokenlist]
        tokenlist = [' ' if re.search("^[+-]?[0-9]+\.[0-9]",token) else token for token in tokenlist] # remove any weird numbers

        # make one string from split list
        proc_text = ' '.join(tokenlist)

        if len(proc_text) > 110:
            return self.split_sentence(proc_text.split(), splits=10)
        else:
            return proc_text

        print(f'{proc_text}\n')
        # proc_text = [token for token in sent if token.isaplha() or token.isdigit()]

        return proc_text


    def split_sentence(self, sent: list[str], splits: int) -> list[str]:

        chunked_sents = self.get_chunks(sent, chunk_size=7) # create chunks of tokens

        # generate single string for each token chunk
        sents = []
        for tokens in chunked_sents:
            str = ' '.join(tokens)
            sents.append(str)

        return sents


    # grabs chunks from a given array
    def get_chunks(self, tokens: list[str], chunk_size: int) -> list[list[str]]:
        chunked_list = []
        for i in range(0, len(tokens), chunk_size):
            chunked_list.append(tokens[i:i+chunk_size])
        return chunked_list


class Model:
    def __init__(self,model_type: int) -> None:
        pass



# run the data through the model
def text_to_speech(data: list, filename: str, batch_size: int) -> torch.Tensor:

    sample_rate = 16000
    device = torch.device('cpu') # change to gpu or cpu depending on users haardware
    # torch.set_num_threads(4)

    local_file = 'model.pt'
    # collect model if it doesnt exist
    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v2_lj.pt',
                local_file)

    # get text to speech model
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    print(f'\nGenerating Text to Speech from {filename}')

    outputs = []
    batches = get_batches(data, batch_size) # make these tensors and maybe into pytorch dataloader

    for batch in tqdm(batches):
        audio = model.apply_tts(texts=batch, sample_rate=sample_rate)

        for wave in audio:
            outputs.append(audio)

    # make single tensor of all wave forms
    # TODO: tidy and more efficient code needed
    waveform = outputs[0]
    for a in tqdm(outputs):
        waveform = torch.cat((waveform, a)) # axis = 1

    return torch.reshape(waveform, shape=(1, len(waveform)))

# grabs a batches sequentially from a list given a batch size
def get_batches(tokens: list[str], chunk_size: int) -> list[list[str]]:
    chunked_list = []

    for i in range(0, len(tokens), chunk_size):
        chunked_list.append(tokens[i:i+chunk_size])

    return chunked_list


def save_audio(waveform: torch.Tensor, path: str) -> None:
    # save audio of pdf
    # audio_paths = model.save_wav(texts=data, sample_rate=sample_rate)
    try:
        torchaudio.save(path, waveform, sample_rate=8000)
        print(f'\nfile saved as {path}')
    except:
        print('save failed..')
        quit()

'''
def text_to_speech(data: list, filename: str, batch_size: int) -> torch.Tensor:
    language = 'en'
    speaker = 'lj_16khz'
    device = torch.device('cpu')

    model, symbols, sample_rate, example_text, apply_tts = \
            torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts',
                    language=language, speaker=speaker)

    model.to(device)

    print(f'\nGenerating Text to Speech from {filename}')

    outputs = []
    batches = get_batches(data, batch_size)

    for sents in tqdm(batches):
        audio = apply_tts(texts=sents,
                model=model,
                sample_rate=sample_rate,
                symbols=symbols,
                device=device)

        # audio = model.apply_tts(texts=sents, sample_rate=sample_rate)
        outputs.append(audio)

    waveform = outputs[0]
    for a in tqdm(outputs):
        waveform = torch.cat((waveform, a)) # axis = 1

    return torch.reshape(waveform, shape=(1, len(waveform)))
    # model.save_wav(texts=waveform, sample_rate=sample_rate)
'''

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

            # run the preprocessed data through the model
            waveform = text_to_speech(data=data, filename=file.filename, batch_size=10)

            # use pdf name as audio files name
            save_audio(waveform=waveform, path=file.audio_path)
    else:
        print('No file')

if __name__ == "__main__":
    main()

