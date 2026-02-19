from core.metrics.masked_metrics import masked_loss, masked_accuracy
from core.preprocessing.dataset import preprocess_data
from core.preprocessing.vectorizers import Vectorizers
from core.model.transformer import Transformer


def train_model(config):
    vectorizers = Vectorizers(config)
    encoder_inputs, decoder_inputs, decoder_targets = preprocess_data(
        config["path"]["data_path"], vectorizers
    )
    model = Transformer(config["model"])
    model.compile(
        optimizer="adam",
        loss=masked_loss,
        metrics=[masked_accuracy],
    )
    model.fit(
        [encoder_inputs, decoder_inputs],
        decoder_targets,
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
    )

    return model, vectorizers
