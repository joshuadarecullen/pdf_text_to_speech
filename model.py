import os
import torch
import torchaudio
from tqdm import tqdm # for loop progress bar

class Model:
    def __init__(self, model_type: int, sample_rate: int) -> None:
        self.model = self.set_model(model_type)
        self.sample_rate = sample_rate
        self.device = torch.device('cpu') # change to gpu or cpu depending on users haardware

    def set_model(self, model_type: int) -> torch.package:
        local_file = 'model.pt'
        # collect model if it doesnt exist
        if not os.path.isfile(local_file):
            torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v2_lj.pt',
                    local_file)

        # get text to speech model
        model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        # print(type(model))

        return model

    # run the data through the model
    def text_to_speech(self, data: list, \
            filename: str, batch_size: int) -> torch.Tensor:

        # torch.set_num_threads(4)

        self.model.to(self.device)

        print(f'\nGenerating audio from {filename}')

        outputs = []
        batches = self.get_batches(data, batch_size) # make these tensors and maybe into pytorch dataloader

        for batch in tqdm(batches):
            audio = self.model.apply_tts(texts=batch, sample_rate=self.sample_rate)

            for wave in audio:
                outputs.append(wave)
        print(f'\nText to speech processing complete... \n')

        # make single tensor of all wave forms
        # TODO: tidy and more efficient code needed
        print('Generating one wave...')
        waveform = outputs[0]
        for a in tqdm(outputs):
            waveform = torch.cat((waveform, a)) # axis = 1

        return torch.reshape(waveform, shape=(1, len(waveform)))

    # grabs a batches sequentially from a list given a batch size
    def get_batches(self, data: list[str], chunk_size: int) -> list[list[str]]:
        chunked_list = []

        for i in range(0, len(data), chunk_size):
            chunked_list.append(data[i:i+chunk_size])

        return chunked_list


    # save audio of pdf
    def save_audio(self, waveform: torch.Tensor, path: str) -> None:
        # audio_paths = model.save_wav(texts=data, sample_rate=sample_rate)
        try:
            torchaudio.save(path, waveform, sample_rate=self.sample_rate)
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

