import operator
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import random
from functools import partial

from typing import Optional, Callable

import optax
import ray
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from loader import TextLoader

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)

class MemAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied and persistant vectors."""

    def __init__(self,
                 mem_vectors: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.mem_vectors = mem_vectors

    def __call__(
            self,
            query: jnp.ndarray,
            key: Optional[jnp.ndarray] = None,
            value: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        query = layer_norm(query)

        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        query_heads = self._linear_projection(query, self.query_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        sqrt_key_size = np.sqrt(self.key_size, dtype=key.dtype)
        query_heads = query_heads / sqrt_key_size

        mem_k = hk.get_parameter("mem_k", [self.mem_vectors, self.num_heads, self.key_size], query.dtype,
                                 init=self.w_init)
        mem_v = hk.get_parameter("mem_v", [self.mem_vectors, self.num_heads, self.value_size], query.dtype,
                                 init=self.w_init)

        mem_k_logit = jnp.einsum("bthd,mhd->bhtm", query_heads, mem_k * 64)

        attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)
        attention_logits -= 1e10 * (1. - mask)

        full_logits = jnp.concatenate((attention_logits, mem_k_logit), axis=-1)
        attention_weights = jax.nn.softmax(full_logits)

        context_weights = attention_weights[:, :, :, :-self.mem_vectors]
        mem_weights = attention_weights[:, :, :, -self.mem_vectors:]

        context_attention = jnp.einsum("bhtT,bThd->bthd", context_weights, value_heads)
        mem_attention = jnp.einsum("bhtm,mhd->bthd", mem_weights, mem_v * 64)

        attention = context_attention + mem_attention

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*query.shape[:2], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attention_vec)


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class ReversibleLayer(object):
    def __init__(
            self,
            layer_init: Callable,
            layer: int,
            data: jnp.ndarray,
            optimizer: optax.GradientTransformation
    ):
        self.layer = layer

        def forward(x):
            f, g = layer_init()

            hidden = x.shape[-1]
            x1 = x[:, :, :hidden // 2]
            x2 = x[:, :, hidden // 2:]

            y1 = f(x2) + x1
            y2 = g(y1) + x2
            return jnp.concatenate((y1, y2), axis=-1)

        def reverse(y):
            f, g = layer_init()

            hidden = y.shape[-1]
            y1 = y[:, :, :hidden // 2]
            y2 = y[:, :, hidden // 2:]

            x2 = y2 - g(y1)
            x1 = y1 - f(x2)
            return jnp.concatenate((x1, x2), axis=-1)

        @functools.partial(jax.jit)
        def opt(state):
            def grad_to_cpu(x):
                return jax.device_put(x, device=jax.devices("cpu")) / state['grad_count']

            grad_cpu = jax.tree_map(grad_to_cpu, state['grad_acc'])

            updates, opt_state = optimizer.update(grad_cpu, state['opt_state'])
            state['params'] = optax.apply_updates(state['params'], updates)

            state['grad_acc'] = jax.tree_map(jnp.zeros_like, state['grad_acc'])
            state['grad_count'] = np.array(0)
            return state

        self.forward_fn = hk.transform(forward)
        self.reverse_fn = hk.transform(reverse)

        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.jit, static_argnums=0)
        def init_fn(master_rng, data):
            out_rng, init_rng = jax.random.split(master_rng)
            params = self.forward_fn.init(init_rng, data)

            # place optimizer state on CPU
            opt_state = jax.tree_map(partial(jax.device_put, device=jax.devices("cpu")), optimizer.init(params))

            return dict(
                step=np.array(0),
                rng=out_rng,
                opt_state=opt_state,
                grad_acc=jax.tree_map(jnp.zeros_like, params),
                grad_count=np.array(0),
                params=params)

        @functools.partial(jax.jit)
        def forward_fn(x, state):
            params = state['params']
            out = self.forward_fn.apply(params, None, x)
            return out

        @functools.partial(jax.jit)
        def reverse_fn(y_dy, state):
            params = state['params']
            acc = state['grad_acc']

            y, dy = y_dy
            reconstr_x = self.reverse_fn.apply(params, None, y)

            _, vjpfun = jax.vjp(self.forward_fn.apply, params, None, reconstr_x)
            weights_grad, _, x_grad = vjpfun(dy)

            state['grad_acc'] = jax.tree_multimap(operator.add, acc, weights_grad)
            state['grad_count'] = state['grad_count'] + 1
            return (reconstr_x, x_grad), state

        self.state = init_fn(master_rng, data)
        self.forward = forward_fn
        self.forward(data, self.state)

        self.reverse = reverse_fn
        self.reverse((data, jnp.zeros_like(data)), self.state)

        self.opt = opt
        self.opt(self.state)

    def forward(self, h):
        return self.forward(h, self.state)

    def backward(self, y_dy):
        x_dx, new_state = self.reverse(y_dy, self.state)
        self.state = new_state
        return x_dx

    def opt(self):
        self.state = self.opt(self.state)


@ray.remote(num_gpus=0.01, num_cpus=0.01)
class EmbeddingLayer(object):
    def __init__(self, obs, vocab: int, d_model: int, optimizer: optax.GradientTransformation):
        self.vocab = vocab
        self.d_model = d_model

        def embed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            return hk.Embed(vocab, d_model, w_init=embed_init, name="embedding")(x)

        def debed_forward(x):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            embed = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding").embeddings

            return x @ embed

        def debed_loss(x, target):
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            embed = hk.Embed(vocab, d_model, w_init=embed_init, name="embedding").embeddings

            logits = x @ embed

            target_onehot = jax.nn.one_hot(target, vocab)
            loss = -jnp.sum(target_onehot * jax.nn.log_softmax(logits), axis=-1)
            loss = jnp.mean(loss)

            return loss

        @functools.partial(jax.jit)
        def opt(state):
            def grad_to_cpu(x):
                return jax.device_put(x, device=jax.devices("cpu")) / state['grad_count']

            grad_cpu = jax.tree_map(grad_to_cpu, state['grad_acc'])

            updates, opt_state = optimizer.update(grad_cpu, state['opt_state'])
            state['params'] = optax.apply_updates(state['params'], updates)

            state['grad_acc'] = jax.tree_map(jnp.zeros_like, state['grad_acc'])
            state['grad_count'] = np.array(0)
            return state

        self.embed_fwd_fn = hk.transform(embed_forward)
        self.debed_fwd_fn = hk.transform(debed_forward)
        self.debed_loss_fn = hk.transform(debed_loss)

        master_rng = jax.random.PRNGKey(random.getrandbits(32))

        @functools.partial(jax.jit, static_argnums=0)
        def init_fn(master_rng, data):
            out_rng, init_rng = jax.random.split(master_rng)
            params = self.embed_fwd_fn.init(init_rng, data)

            # place optimizer state on CPU
            opt_state = jax.tree_map(partial(jax.device_put, device=jax.devices("cpu")), optimizer.init(params))

            return dict(
                step=np.array(0),
                rng=out_rng,
                opt_state=opt_state,
                grad_acc=jax.tree_map(jnp.zeros_like, params),
                grad_count=np.array(0),
                params=params)

        @functools.partial(jax.jit)
        def embed_fwd_fn(obs, state):
            params = state['params']
            out = self.embed_fwd_fn.apply(params, None, obs)

            return out

        @functools.partial(jax.jit)
        def embed_grad_fn(obs, y_dy, state):
            params = state['params']
            acc = state['grad_acc']

            y, dy = y_dy

            y_new, vjpfun = jax.vjp(self.embed_fwd_fn.apply, params, None, obs)
            weights_grad, _, _ = vjpfun(dy)

            diff = jnp.square(y - y_new).mean()

            state['grad_acc'] = jax.tree_multimap(operator.add, acc, weights_grad)
            state['grad_count'] = state['grad_count'] + 1

            return diff, state

        @functools.partial(jax.jit)
        def debed_fwd_fn(target, state):
            params = state['params']
            out = self.debed_fwd_fn.apply(params, None, target)

            return out

        @functools.partial(jax.jit)
        def debed_grad_fn(hidden, target, state):
            params = state['params']
            acc = state['grad_acc']

            loss, vjpfun = jax.vjp(self.debed_loss_fn.apply, params, None, hidden, target)
            weights_grad, _, x_grad, _ = vjpfun(np.ones((), dtype=hidden.dtype))

            state['grad_acc'] = jax.tree_multimap(operator.add, acc, weights_grad)
            state['grad_count'] = state['grad_count'] + 1

            return hidden, x_grad, loss, state

        # we call all the functions here to trigger jit at init
        self.state = init_fn(master_rng, obs)
        self.embed_fwd = embed_fwd_fn
        e = self.embed_fwd(obs, self.state)

        self.embed_grad = embed_grad_fn
        self.embed_grad(obs, (e, e), self.state)

        self.debed_fwd = debed_fwd_fn
        self.debed_fwd(e, self.state)

        self.debed_grad = debed_grad_fn
        self.debed_grad(e, np.ones_like(e).mean(axis=-1), self.state)

        self.opt = opt
        self.opt(self.state)
        
    def embed_forward(self, obs):
        return self.embed_fwd(obs, self.state)

    def embed_grad(self, obs, y_dy):
        diff, state = self.embed_grad(obs, y_dy, self.state)
        self.state = state

        return diff

    def debed_forward(self, h):
        return self.debed_fwd(h, self.state)

    @ray.method(num_returns=2)
    def debed_grad(self, h, targets):
        hidden, x_grad, loss, state = self.debed_grad(h, targets, self.state)
        self.state = state

        return (hidden, x_grad), loss

    def opt(self):
        self.state = self.opt(self.state)

    def __repr__(self):
        return "EmbeddingLayer"


train_dataset = TextLoader("data/enwik8", batchsize=16, sample_size=128, length=90000000)
data = train_dataset.get_samples()
optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99))

ray.init()

embedding_actor = EmbeddingLayer.remote(data["obs"], 256, 256, optimizer)

embedding = embedding_actor.embed_forward.remote(data["obs"])

layers = []

for i in range(12):
    def layer_init():
        f = MemAttention(
            num_heads=4,
            key_size=32,
            w_init_scale=2/12,
            name=f'l{i}_f',
            mem_vectors=512
        )

        g = MemAttention(
            num_heads=4,
            key_size=32,
            w_init_scale=2/12,
            name=f'l{i}_g',
            mem_vectors=512
        )

        return f, g
    layers.append(ReversibleLayer.remote(layer_init, i, embedding, optimizer))

dbg = False

def train_sample():
    data = train_dataset.get_samples()

    x = embedding_actor.embed_forward.remote(data["obs"])
    ray.wait([x])

    if dbg:
        fwd_activations = []
        bwd_activations = []

    for l in layers:
        if dbg:
            fwd_activations.append(x)
        x = l.forward.remote(x)
        ray.wait([x])

    y_dy, loss = embedding_actor.debed_grad.remote(x, data["target"])
    ray.wait([y_dy])

    for l in reversed(layers):
        y_dy = l.backward.remote(y_dy)
        ray.wait([y_dy])
        if dbg:
            bwd_activations.append(y_dy)

    error = embedding_actor.embed_grad.remote(data["obs"], y_dy)
    ray.wait([error])

    print(ray.get(loss))

    opts = [embedding_actor.opt.remote()]

    for l in layers:
        opts.append(l.opt.remote())

    ray.wait(opts, num_returns=len(opts))

    if dbg:
        fwd_activations = ray.get(fwd_activations)
        bwd_activations_grad = ray.get(bwd_activations)

        bwd_activations = [i[0] for i in bwd_activations_grad]
        bwd_activations.reverse()

        for f, b in zip(fwd_activations, bwd_activations):
            assert jnp.allclose(f, b, rtol=1e-4, atol=1e-4)

while True:
    train_sample()

ray.shutdown()