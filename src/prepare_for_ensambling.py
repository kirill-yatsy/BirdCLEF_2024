import onnxruntime as nxrun
from src.config import CONFIG
from src.data.get_data_loaders import get_data_loaders
from tqdm import tqdm
import torch
from src.augmentations.get_spectrogram_augmentations import MonoToThreeChannel, NoOpTransform, NormalizeData, RandomGain, RandomGainTransition, RandomGaussianNoise, RandomGaussianSNR, RandomLowPassFilter, RandomPitchShift
from torchvision.transforms import v2
import pandas as pd
import numpy as np 

providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": torch.cuda.current_device(),
            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
        },
    )
]

FRAME_LENGTH = 5
image_width = 157
audio_len = 32000 * FRAME_LENGTH
hop_length = audio_len // (image_width - 1)

CONFIG.augmentations.sequence = v2.Compose([ 
       
        MonoToThreeChannel(
            sample_rate=32000, 
            n_mels=128,
            n_fft=2048,
            hop_length=hop_length, 
            top_db=80, 
            f_min=0,
            f_max=16000, 
            n_mfcc=20,
            n_chroma=12
        ),
        NormalizeData(), 
    ])

CONFIG.data_processing.frame_length = FRAME_LENGTH
CONFIG.data_processing.hop_length = hop_length

def collect_predictions():
    # fix random seed
    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)

    classMapperDF = pd.read_csv(f"data/processed/fine_tune_mapper.csv")
 
    _, train_loader, val_loader = get_data_loaders(CONFIG)

    new_rows = []

    model = nxrun.InferenceSession(f"{CONFIG.fine_tune_path}.onnx", providers=providers)
    # model = nxrun.InferenceSession(CONFIG.fine_tune_path)
    
    iterations = 3

    for iteration in range(iterations):
        for i, (_, y, _, x) in tqdm(enumerate(train_loader), total=len(train_loader)): 
            y_hat = model.run(None, {"input": x.numpy()})[0]
            y_hat =  np.exp(y_hat) / np.sum(np.exp(y_hat), axis=1)[:, None]
            for id in range(0, len(y_hat)):
                new_rows.append(np.concatenate(([int(y[id])], y_hat[id])))  
    
    # create dumb of stacked_y_hat and stacked_y
    
    # create dataframe with stacked_y_hat and stacked_y
    df = pd.DataFrame(new_rows, columns=np.concatenate((["y"], classMapperDF["species"] )))
    df["y"] = df["y"].astype(int)
    df.to_csv(f"{CONFIG.fine_tune_path}.train.csv", index=False)

    new_rows = []
    
    for iteration in range(iterations):
        for i, (_, y, _, x) in tqdm(enumerate(val_loader), total=len(val_loader)): 
            y_hat = model.run(None, {"input": x.numpy()})[0]
            y_hat =  np.exp(y_hat) / np.sum(np.exp(y_hat), axis=1)[:, None]
            for id in range(0, len(y_hat)):
                new_rows.append(np.concatenate(([int(y[id])], y_hat[id])))
    
    # create dataframe with stacked_y_hat and stacked_y
    df = pd.DataFrame(new_rows, columns=np.concatenate((["y"], classMapperDF["species"] )))
    df["y"] = df["y"].astype(int)
    df.to_csv(f"{CONFIG.fine_tune_path}.val.csv", index=False)


if __name__ == "__main__":
    collect_predictions()
