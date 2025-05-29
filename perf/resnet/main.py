import argparse
import time
import os
import json

from functools import partial
from typing import Any, Tuple
from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1.5."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)

ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)

ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)

ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)

ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)

ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)


def loss_fn(p, x, y):
    y_pred, _ = model.apply(p, x, train=True, mutable=["batch_stats"])
    return jnp.mean((y_pred - y) ** 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=list, default=[1, 4, 32, 128])
    parser.add_argument("--model-size", type=list, default=[18, 34, 50, 101, 152])
    args = parser.parse_args()

    timings = dict()

    for model_size in args.model_size:
        if model_size == 18:
            model = ResNet18
        elif model_size == 34:
            model = ResNet34
        elif model_size == 50:
            model = ResNet50
        elif model_size == 101:
            model = ResNet101
        elif model_size == 152:
            model = ResNet152
        elif model_size == 200:
            model = ResNet200

        model = model(num_classes=1000)

        timings[model_size] = dict()

        for b in args.batch_size:
            print(f"batch_size={b}")

            x = jnp.ones((b, 224, 224, 3), jnp.float32)
            y_true = jnp.ones((b, 1000), jnp.float32)  # Dummy true labels
            params = model.init(random.PRNGKey(0), x, train=False)
            param_count = sum(x.size for x in jax.tree.leaves(params))

            print(f"Param count: {param_count}")

            apply_fn_compiled = (
                jax.jit(partial(model.apply, train=False)).lower(params, x).compile()
            )
            grad_fn_compiled = (
                jax.jit(jax.grad(loss_fn)).lower(params, x, y_true).compile()
            )

            best_forward_timing = np.inf
            for i in range(100):
                t1 = time.time()
                apply_fn_compiled(params, x).block_until_ready()
                t2 = time.time()
                best_forward_timing = min(best_forward_timing, t2 - t1)

            # Backward pass timing
            best_backward_timing = np.inf
            for i in range(100):
                t1 = time.time()
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready(), grad_fn_compiled(params, x, y_true)
                )
                t2 = time.time()
                best_backward_timing = min(best_backward_timing, t2 - t1)

            timings[model_size][b] = {
                "forward": best_forward_timing,
                "backward": best_backward_timing,
            }
            print(f"Best forward timing: {best_forward_timing:.5f} s")
            print(f"Best backward timing: {best_backward_timing:.5f} s")

    print(timings)

    results_path = os.path.join(os.path.dirname(__file__), "../results/resnet/")
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, "jax.json"), "w") as f:
        json.dump(timings, f, indent=4)
