"""Entrenamiento compilado: el bucle de optimización entero corre dentro de XLA
(jax.lax.scan sobre bloques de pasos), sin despacho Python por paso."""
import jax, jax.numpy as jnp, optax


def fit(loss, p, n, steps, seed, lr=2e-3, batch=256, chunk=500):
    """loss(p, idx) -> escalar; idx = índices de batch muestreados dentro de XLA."""
    opt = optax.adam(lr)
    st = opt.init(p)

    def one(carry, k):
        p, st = carry
        idx = jax.random.randint(k, (batch,), 0, n)
        l, g = jax.value_and_grad(loss)(p, idx)
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st), l

    @jax.jit
    def block(p, st, keys):
        (p, st), ls = jax.lax.scan(one, (p, st), keys)
        return p, st, ls[-1]

    keys = jax.random.split(jax.random.PRNGKey(seed), steps)
    l = None
    for i in range(0, steps, chunk):
        p, st, l = block(p, st, keys[i:i + chunk])
    return p, float(l)


def fit_many(loss, ps, n, steps, seeds, lr=2e-3, batch=256, chunk=500):
    """Entrena S réplicas (semillas) en paralelo con jax.vmap dentro del mismo
    programa XLA. ps: pytree con eje líder S. Devuelve params con eje S."""
    opt = optax.adam(lr)
    st = jax.vmap(opt.init)(ps)

    def one(carry, ks):
        p, st = carry

        def single(p, st, k):
            idx = jax.random.randint(k, (batch,), 0, n)
            l, g = jax.value_and_grad(loss)(p, idx)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l

        p, st, l = jax.vmap(single)(p, st, ks)
        return (p, st), l

    @jax.jit
    def block(p, st, keys):
        (p, st), ls = jax.lax.scan(one, (p, st), keys)
        return p, st, ls[-1]

    keys = jnp.stack([jax.random.split(jax.random.PRNGKey(s), steps) for s in seeds], 1)
    l = None
    for i in range(0, steps, chunk):
        ps, st, l = block(ps, st, keys[i:i + chunk])
    return ps, [float(x) for x in l]
