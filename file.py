import os
import re

from num2words import num2words
from tqdm import tqdm # for loop progress bar
from PyPDF2 import PdfReader # pdf reader module
from nltk.tokenize import sent_tokenize as sent_tokeniser # tokeniser, unused atm
from nltk.tokenize import word_tokenize as word_tokeniser # tokeniser, unused atm


class File:
    def __init__(self, path: str, audio_type:str) -> None:
        self.INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .txt file."
        self.INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."
        self.validate_file(path=path)
        self.process_paths(audio_type)

    def process_paths(self, audio_type: str) -> None:

        if re.search('/', self.path):
            split_path = self.path.split('/') #  split at '/'
            self.filename = split_path[-1] # collect filename
            reduce_filename = re.sub('\.', '-', str(self.filename)) # remove '.'
            audio_path = '/'.join(split_path[1:-1])
            self.audio_path = audio_path + f"/{reduce_filename}.{audio_type}"
        else:
            self.filename = self.path # collect filename
            reduce_filename = re.sub('\.', '-', str(self.filename)) # remove '.'
            self.audio_path = f"{reduce_filename}.{audio_type}"

        print(self.audio_path)


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
        print('\nReading PDF...')
        self.reader = PdfReader(self.path)

    # pre process the data collected from the pdf
    def pre_process_pdf(self) -> str:
        all_pages = [] # collect all the text

        print('\nProcessing the text from each page of the pdf')
        for i in tqdm(range(self.reader.numPages)):
            page = self.reader.pages[i]
            text = self.process_text(page.extractText())
            all_pages += text if isinstance(text, list) else all_pages.append(text)

        return all_pages

    # process text from each page
    def process_text(self, page: str) -> str or list[str]:
        proc_text = re.sub('\n', ' ', page) # remove new line breaks

        # TODO: change brackets and quotes to their spoke word
        # proc_text = if re.sub(r'^\"[a-zA-Z\d]"$', token)

        proc_sents = []
        for sent in sent_tokeniser(proc_text):
            # print(sent)
            proc_sent = [str(token) for token in word_tokeniser(sent) if token not in '\/:[]()']

            # pp -> page
            proc_sent = ['page' if re.search(r'^pp$', token) else token for token in proc_sent]

            # deal with digits in the string and convert to spoken form
            proc_sent = [self.num_to_word(token) if token.isdigit() else token for token in proc_sent]
            proc_sent = [' to '.join(self.num_to_word(token.split('–'))) if re.search(r'–', token) else token for token in proc_sent]
            proc_sent = [' to '.join(self.num_to_word(token.split('-'))) if re.search(r'^\d*[-]\da+$', token) else token for token in proc_sent]

            # deal with '-'
            proc_sent = [' '.join(self.check_for_page_ref(token.split('-'))) if re.search(r'\-', token) else token for token in proc_sent]

            # deal with '4B' type scenarios
            # proc_sent = [' '.join(self.num_to_word(re.sub('^[a-z][A-Z]*$','', token))) + f" {re.search(r'^[a-z][A-Z]*$', token)}" if re.search(r"^\d*[a-zA-Z]+$",token) else token for token in proc_sent]

            # change ordinals digits to word rep
            # proc_sent = [re.sub('\-', ' ', num2words(token[:-2], to = 'ordinal_num'))
            #         if (token.endswith(("nd","st","th")) and token[:-2].isdigit())
            #         else token
            #         for token in proc_tokens]

            # join all tokens to recreate sentence
            proc_sents.append(' '.join(proc_sent))

            for token in proc_sents:
                print(token)


        # create whole page again
        proc_text = ' '.join(proc_sents)
        print(f'{proc_text}\n')

        # if split page into required input chunks for model
        if len(proc_text) > 110:
            return self.split_sentence(proc_text.split(), splits=10)
        else:
            return proc_text

    # deals with '-' and deals with the token whether it contains digits or charactres
    def check_for_page_ref(self, tokens: list[str]) -> list[str]:

        # check whether all are digits
        all_digits = True
        for token in tokens:
            if not token.isdigit():
                all_digits = False

        if all_digits:
            word_list = []
            for token in tokens:
                word_list += self.num_to_word(token)
        else:
            word_list = []
            for token in tokens:
                if token.isdigit():
                    word_list += self.num_to_word(token)
                else:
                    word_list.append(token)

        return word_list

    # convert number to word representation
    def num_to_word(self, token: str or list) -> str or list:

        # if a string is passed in
        if isinstance(token, str):
            token = num2words(token)
            if re.search('\-', token): #  num2words places '-' between word results
                token = re.sub('\-', ' ', token) #  remove -
                tokens = token.split() #  split at space eg token = 'twenty four'
                return token
            else:
                return token
        else:
            num_words = []
            # multiply digits in a list
            for t in token:
                tokens = num2words(t)
                if re.search('\-', tokens):
                    tokens = re.sub('\-', ' ', tokens)
                    num_words.append(tokens)
                else:
                    num_words += [tokens]
            return num_words


    # split sentence to be 140 per sentence to fit model input
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
