from src.config import CONFIG
from src.data.get_data_loaders import get_data_loaders
from src.model.BirdCleffModel import BirdCleffModel
import torch.onnx
from torchsummary import summary
import timm


def export_to_kaggle():
    df, train_loader, val_loader = get_data_loaders(CONFIG)
    iterator = iter(train_loader)
    _, target, _, spec = next(iterator)
    model_wrapper = BirdCleffModel(df, 926)
    model_wrapper.load_from_checkpoint(
        "checkpoints/efficientnet_b1/model-epoch=07-val_loss=2.01.ckpt"
    )
    model_wrapper.to_onnx("model2.onnx", spec)


def export_to_onnx():
    df, train_loader, val_loader = get_data_loaders(CONFIG)
    model = BirdCleffModel.load_from_checkpoint(
        "checkpoints/efficientnet_b0/model-epoch=08-val_loss=3.50.ckpt",
        df=df,
        num_classes=182,
    )
    loader = iter(train_loader)
    _, target, _, spec = next(loader)
    print(spec.shape)

    model.eval()
    dummy_input = torch.randn(1,  spec.shape[1],  spec.shape[2], spec.shape[3])

    print(summary(model, torch.randn( spec.shape[1],  spec.shape[2], spec.shape[3]).shape))
    
    
    model.to("cpu")
    # # Export the model with dynamic axes for batch size
    # torch.onnx.export(
    #     model, dummy_input, "model.onnx", dynamic_axes={"input": {0: "batch_size"}}
    # )

    input_name = "input"
    output_name = "output"

    torch.onnx.export(model.model, dummy_input, "model2.onnx", 
                  input_names=[input_name],
                  output_names=[output_name],
                  dynamic_axes={input_name: {0: "batch_size"}, output_name: {0: "batch_size"}})


def export_to_jit():
    df, train_loader, val_loader = get_data_loaders(CONFIG)
    model = BirdCleffModel.load_from_checkpoint(
        "checkpoints/efficientnet_b1/model-epoch=02-val_loss=2.64.ckpt",
        df=df,
        num_classes=182,
    )

    loader = iter(train_loader)
    _, target, _, spec = next(loader)
    print(spec.shape)


    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1,  spec.shape[1],  spec.shape[2], spec.shape[3])
    traced_model = torch.jit.trace(model.model, dummy_input)
    traced_model.save("model.pt")
    model = timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=True,
        num_classes=182,
        global_pool="avg",
        in_chans=3,
    )

    model = model.eval()
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("test_www.pt")

if __name__ == "__main__":
    export_to_onnx()
    export_to_jit()
    # small_image = torch.randn(1, 3, 224, 224)
    # big_image = torch.randn(1, 3, 3333, 2324)
    # df, train_loader, val_loader = get_data_loaders(CONFIG)
    # model = BirdCleffModel.load_from_checkpoint(
    #     "checkpoints/efficientnet_b0/model-epoch=08-val_loss=3.50.ckpt",
    #     df=df,
    #     num_classes=182,
    # )

    # model(small_image.cuda())
    # model(big_image.cuda())
