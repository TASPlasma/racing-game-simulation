# file for training transformer on data
from data_generator import DataGenerator
from preprocessor import Preprocessor
import torch
from tqdm import tqdm
import jax
import numpy as np
import jax.numpy as jnp
import jax.nn as nn
import equinox as eqx
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from transformer.transformer import Transformer
from transformer.config import Config
from jaxtyping import Array, Float, Int, PyTree
import optax
from typing import Callable, Tuple

print(f'{jax.devices()}')


class GeneratorDataset(IterableDataset):
    def __init__(self, data_generator: DataGenerator, preprocessor: Preprocessor, num_samples: int):
        self.data_generator = data_generator
        self.preprocessor = preprocessor
        self.num_samples = num_samples

    def __iter__(self):
        for datum in self.data_generator():
            inputs = datum['inputs']
            pos = inputs['pos']
            vel = inputs['vel']
            dir = inputs['dir']
            vertices = inputs['course_vertices']

            model_input = self.preprocessor.encode_input(
                pos, vel, dir, vertices)
            label_seq = datum['label']['label']
            # prepend SOS token to label_seq
            label_seq = preprocessor.prepend_sos(label_seq, 0)
            label_one_hot = self.preprocessor.one_hot_encoded([label_seq])
            model_input = np.array(model_input)
            label_one_hot = np.array(label_one_hot)
            model_input = torch.from_numpy(model_input)
            label_one_hot = torch.from_numpy(
                label_one_hot).squeeze(0)

            yield model_input, label_one_hot

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


@dataclass
class Hyperparameters:
    """
    lr: learning rate
    num_epochs: number of epochs
    num_steps: number of training steps
    print_every: int for printing every x steps
    """
    lr: float = 0.001
    num_epochs: int = 5
    num_steps: int = 100
    print_every: int = 10


def cross_entropy_loss(logits, targets):
    log_probs = nn.log_softmax(logits, axis=-1)
    return -jnp.sum(targets * log_probs, axis=-1)


def loss(
    model: Transformer,
    x: Float[Array, "batch seq_len input_dim"],
    y: Float[Array, "batch seq_len+1 num_classes"],
    loss_crit: Callable,
):
    """
    Compute the loss for the Transformer model

    :param model: The Transformer model
    :param x: Input sequences
    :param y: One-hot encoded target sequences, already has SOS token
    :param loss_crit: Loss criterion function
    :return: Scalar loss value
    """
    y_in = y[:, :-1, :]  # (batch, seq_len, num_classes)

    # shape: (batch, seq_len, num_classes)
    logits = jax.vmap(model)(x, y_in)

    # y[:, 1:, :] : (batch, seq_len, num_classes)
    # in particular ignores the SOS token
    loss_value = loss_crit(logits, y[:, 1:, :])

    return jnp.mean(loss_value)


def compute_critical_errors(pred_y: jnp.ndarray, true_y: jnp.ndarray) -> int:
    """
    Compute the number of critical errors.
    A critical error occurs when the predicted mean exceeds the true mean.

    :param pred_y: Predicted labels
    :param true_y: True labels
    :return: Number of critical errors
    """
    return jnp.sum(jnp.mean(pred_y, axis=-1) > jnp.mean(true_y, axis=-1))


@eqx.filter_jit
def evaluate(
    model: Transformer,
    testloader: DataLoader,
    loss_crit: Callable
) -> Tuple[float, float, int]:
    """
    Evaluate the model on the test set.

    :param model: The Transformer model
    :param testloader: DataLoader for the test set
    :param loss_fn: Loss function
    :return: Tuple of (average loss, accuracy, number of critical errors)
    """
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    critical_errors = 0

    @eqx.filter_jit
    def metric(model, x, y, loss_crit):
        y_in = y[:, :-1, :]  # (batch, seq_len, num_classes)

        # shape: (batch, seq_len, num_classes)
        logits = jax.vmap(model)(x, y_in)

        # y[:, 1:, :] : (batch, seq_len, num_classes)
        # in particular ignores the SOS token
        loss_value = loss_crit(logits, y[:, 1:, :])

        return jnp.mean(loss_value), logits

    print('uwu did this happen before for loop in evaluate')

    for batch in tqdm(testloader):
        x, y = batch
        x = jnp.array(x)
        y = jnp.array(y)

        # Compute loss using the provided loss function
        batch_loss, logits = metric(model, x, y, loss_crit)
        print(f'{batch_loss, logits}')
        total_loss += batch_loss

        pred_y = jnp.argmax(logits, axis=-1)
        true_y = jnp.argmax(y[:, 1:, :], axis=-1)
        print(f'{true_y}')

        # Compute accuracy
        correct_predictions += jnp.sum(pred_y == true_y)
        total_predictions += pred_y.size

        # Check for critical errors
        critical_errors += compute_critical_errors(pred_y, true_y)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(testloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy, critical_errors


def train(
    loss,
    model: Transformer,
    train_data: DataLoader,
    val_data: DataLoader,
    test_data: DataLoader,
    optim: optax.GradientTransformation,
    loss_crit: Callable,
    hypers: Hyperparameters
) -> Transformer:

    # filter out non-arrays
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: Transformer,
        opt_state: PyTree,
        x: Float[Array, "batch seq_len input_dim"],
        y: Float[Array, "batch seq_len+1 num_classes"]
    ):
        print(f'{x.shape=}, {y.shape=}')
        loss_value, grads = eqx.filter_value_and_grad(
            lambda m: loss(m, x, y, loss_crit))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def infinite_trainloader():
        while True:
            yield from train_data

    for step, (x, y) in zip(range(hypers.num_steps), infinite_trainloader()):
        x = x.numpy()
        y = y.numpy()
        x = jnp.array(x)
        y = jnp.array(y)
        print(f'211 train.py: {x.shape}, {y.shape}')
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % hypers.print_every == 0) or (step == hypers.num_steps - 1):
            val_loss, val_accuracy, critical_errors = evaluate(
                model, val_data, loss_crit)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={val_loss.item()}, test_accuracy={val_accuracy.item()}"
                f"critical errors: {critical_errors.item()}"
            )

    final_test_loss, final_test_accuracy, final_critical_errors = evaluate(
        model, test_data, loss_crit)
    print(f"Final Test Results: loss={final_test_loss.item()}, "
          f"accuracy={final_test_accuracy.item()}, "
          f"critical errors={final_critical_errors}")

    return model


train_num_samples = 1000
train_generator = DataGenerator(train_num_samples)
preprocessor = Preprocessor()

train_data = GeneratorDataset(train_generator, preprocessor, train_num_samples)
train_loader = DataLoader(train_data, batch_size=32,
                          collate_fn=collate_fn)

val_num_samples = 50
val_generator = DataGenerator(val_num_samples)

val_data = GeneratorDataset(val_generator, preprocessor, val_num_samples)
val_loader = DataLoader(val_data, batch_size=32,
                        collate_fn=collate_fn)

test_num_samples = 50
test_generator = DataGenerator(test_num_samples)

test_data = GeneratorDataset(test_generator, preprocessor, test_num_samples)
test_loader = DataLoader(test_data, batch_size=32,
                         collate_fn=collate_fn)

# config, hyperparameters and model initialization
cfg = Config(sizes=[4, 20, 8], ff_sizes=[20, 8],
             d_k=4, num_heads=2, num_layers=2)
key = jax.random.PRNGKey(cfg.seed)
key, subkey = jax.random.split(key, 2)
hypers = Hyperparameters(num_epochs=1, num_steps=2, print_every=1)

model = Transformer(cfg, key)  # model initialization

# opmtimizer
optim = optax.adamw(hypers.lr)


# trained_model = train(
#     eqx.filter_jit(loss),
#     model,
#     train_loader,
#     train_loader,
#     train_loader,
#     optim,
#     cross_entropy_loss,
#     hypers
# )
