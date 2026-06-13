"""Correctness-first screened Coulomb backend built on jaxFMM hierarchies.

This module uses the existing adaptive hierarchy and interaction lists from
jaxFMM, but evaluates a pure Yukawa kernel. Well-separated source boxes are
compressed by tensor-product Chebyshev interpolation; neighboring boxes are
computed exactly by direct P2P. The implementation is independent from the
Laplace-specific ``jaxfmm.eval_potential`` path.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from opendust_jax.direct import _validate_inputs
from opendust_jax.fmm_poisson import _import_jaxfmm


def build_yukawa_tree(
    positions: jax.Array,
    n_max: int = 128,
    theta: float = 0.77,
    p: int = 4,
) -> dict[str, Any]:
    """Build a hierarchy with debug box lengths needed by the Yukawa backend."""

    positions = jnp.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if n_max <= 0:
        raise ValueError("n_max must be positive.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    if p < 1:
        raise ValueError("p must be at least 1 for Chebyshev interpolation.")

    _, gen_hierarchy = _import_jaxfmm()
    return gen_hierarchy(
        positions,
        N_max=int(n_max),
        theta=float(theta),
        p=int(p),
        debug_info=True,
    )


def _chebyshev_nodes(order: int) -> jax.Array:
    if order == 0:
        return jnp.array([0.0])
    k = jnp.arange(order + 1)
    return jnp.cos(jnp.pi * k / order)


def _tensor_nodes(order: int) -> jax.Array:
    nodes_1d = _chebyshev_nodes(order)
    xx, yy, zz = jnp.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij")
    return jnp.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=1)


@jax.jit
def _lagrange_values_1d(x: jax.Array, nodes: jax.Array) -> jax.Array:
    n = nodes.shape[0]
    values = []
    for i in range(n):
        numerator = jnp.ones_like(x)
        denominator = 1.0
        for j in range(n):
            if i != j:
                numerator = numerator * (x - nodes[j])
                denominator = denominator * (nodes[i] - nodes[j])
        values.append(numerator / denominator)
    return jnp.stack(values, axis=-1)


@jax.jit
def _yukawa_field_from_sources(
    eval_points: jax.Array,
    source_points: jax.Array,
    source_charges: jax.Array,
    kappa: float,
) -> jax.Array:
    diff = eval_points[:, None, :] - source_points[None, :, :]
    r = jnp.linalg.norm(diff, axis=-1)
    inv_r = jnp.where(r == 0.0, 0.0, 1.0 / r)
    screened = jnp.exp(-kappa * r)
    radial = screened * (inv_r**3 + kappa * inv_r**2)
    return jnp.sum(source_charges[None, :, None] * diff * radial[:, :, None], axis=1)


@jax.jit
def _compute_node_charges(
    padded_pts: jax.Array,
    padded_chrgs: jax.Array,
    centers: jax.Array,
    lens: jax.Array,
    local_nodes: jax.Array,
    nodes_1d: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    half_lens = jnp.where(lens == 0.0, 1.0, 0.5 * lens)
    normalized = (padded_pts - centers[:, None, :]) / half_lens[:, None, :]
    lx = _lagrange_values_1d(normalized[..., 0], nodes_1d)
    ly = _lagrange_values_1d(normalized[..., 1], nodes_1d)
    lz = _lagrange_values_1d(normalized[..., 2], nodes_1d)

    nx = nodes_1d.shape[0]
    weights = []
    for node in local_nodes:
        ix = jnp.argmin(jnp.abs(nodes_1d - node[0]))
        iy = jnp.argmin(jnp.abs(nodes_1d - node[1]))
        iz = jnp.argmin(jnp.abs(nodes_1d - node[2]))
        weights.append(
            jnp.take(lx, ix, axis=-1)
            * jnp.take(ly, iy, axis=-1)
            * jnp.take(lz, iz, axis=-1)
        )
    interp = jnp.stack(weights, axis=-1)
    node_charges = jnp.sum(padded_chrgs[..., None] * interp, axis=1)
    node_positions = centers[:, None, :] + half_lens[:, None, :] * local_nodes[None, :, :]
    return node_positions, node_charges


@jax.jit
def _eval_far_field(
    padded_eval_pts: jax.Array,
    node_positions_global: jax.Array,
    node_charges_global: jax.Array,
    mpl_cnct: jax.Array,
    trg_leaf_offset: int,
    src_leaf_offset: int,
    kappa: float,
) -> jax.Array:
    field = jnp.zeros_like(padded_eval_pts)

    def scan_body(buf: jax.Array, pair: jax.Array) -> tuple[jax.Array, None]:
        trg_local = pair[0] - trg_leaf_offset
        src_global = pair[1]
        eval_points = padded_eval_pts[trg_local]
        source_points = node_positions_global[src_global]
        source_charges = node_charges_global[src_global]
        chunk = _yukawa_field_from_sources(eval_points, source_points, source_charges, kappa)
        buf = buf.at[trg_local].add(chunk)
        return buf, None

    field, _ = jax.lax.scan(scan_body, field, mpl_cnct)
    return field


@jax.jit
def _eval_near_field(
    dir_padded_pts: jax.Array,
    dir_padded_chrgs: jax.Array,
    dir_padded_eval_pts: jax.Array,
    dir_cnct: jax.Array,
    kappa: float,
) -> jax.Array:
    field = jnp.zeros_like(dir_padded_eval_pts)

    def scan_body(buf: jax.Array, pair: jax.Array) -> tuple[jax.Array, None]:
        trg = pair[0]
        src = pair[1]
        eval_points = dir_padded_eval_pts[trg]
        source_points = dir_padded_pts[src]
        source_charges = dir_padded_chrgs[src]
        chunk = _yukawa_field_from_sources(eval_points, source_points, source_charges, kappa)
        buf = buf.at[trg].add(chunk)
        return buf, None

    field, _ = jax.lax.scan(scan_body, field, dir_cnct)
    return field


def yukawa_fmm_field(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    eps0: float = 8.85418781762039e-12,
    p: int = 4,
    theta: float = 0.77,
    n_max: int = 128,
    tree: dict[str, Any] | None = None,
) -> jax.Array:
    """Evaluate the free-space Yukawa electric field using the screened backend."""

    positions, charges = _validate_inputs(positions, charges)
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")
    if eps0 <= 0:
        raise ValueError("eps0 must be positive.")
    if tree is None:
        tree = build_yukawa_tree(positions, n_max=n_max, theta=theta, p=p)

    _import_jaxfmm()
    from jaxfmm.hierarchy import handle_padding

    padded = handle_padding(tree["pts"], charges, tree["eval_pts"], tree["idcs"])
    padded_pts, padded_chrgs, padded_eval_pts = padded[:3]
    dir_padded_pts, dir_padded_chrgs, dir_padded_eval_pts = padded[3:]

    order = int(tree.get("p", p))
    nodes_1d = _chebyshev_nodes(order)
    local_nodes = _tensor_nodes(order)
    src_lvl = tree["lvl_info"][-2][1]
    trg_lvl = tree["lvl_info"][-2][0]
    src_leaf_offset = tree["src_ofs"][src_lvl]
    trg_leaf_offset = tree["trg_ofs"][trg_lvl]
    src_next_offset = tree["src_ofs"][src_lvl + 1]

    source_centers = tree["boxcenters"][src_leaf_offset:src_next_offset]
    source_lens = tree["boxlens"][src_leaf_offset:src_next_offset]
    node_positions, node_charges = _compute_node_charges(
        padded_pts,
        padded_chrgs,
        source_centers,
        source_lens,
        local_nodes,
        nodes_1d,
    )

    n_global_src = tree["boxcenters"].shape[0]
    n_nodes = local_nodes.shape[0]
    node_positions_global = jnp.zeros((n_global_src, n_nodes, 3), dtype=positions.dtype)
    node_charges_global = jnp.zeros((n_global_src, n_nodes), dtype=charges.dtype)
    node_positions_global = node_positions_global.at[src_leaf_offset:src_next_offset].set(node_positions)
    node_charges_global = node_charges_global.at[src_leaf_offset:src_next_offset].set(node_charges)

    if tree["mpl_cnct"].size == 0:
        far_leaf = jnp.zeros_like(padded_eval_pts)
    else:
        far_leaf = _eval_far_field(
            padded_eval_pts,
            node_positions_global,
            node_charges_global,
            tree["mpl_cnct"],
            trg_leaf_offset,
            src_leaf_offset,
            kappa,
        )
    if tree["dir_cnct"].size == 0:
        near_leaf = jnp.zeros_like(dir_padded_eval_pts)
    else:
        near_leaf = _eval_near_field(
            dir_padded_pts,
            dir_padded_chrgs,
            dir_padded_eval_pts,
            tree["dir_cnct"],
            kappa,
        )
    far = far_leaf.reshape((-1, 3))[tree["idcs"][1][1]]
    near = near_leaf.reshape((-1, 3))[tree["idcs"][3][1]]
    return (far + near) / (4.0 * jnp.pi * eps0)


def yukawa_fmm_forces(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    eps0: float = 8.85418781762039e-12,
    p: int = 4,
    theta: float = 0.77,
    n_max: int = 128,
    tree: dict[str, Any] | None = None,
) -> jax.Array:
    """Evaluate Yukawa forces on each particle in newtons."""

    positions, charges = _validate_inputs(positions, charges)
    field = yukawa_fmm_field(
        positions,
        charges,
        kappa=kappa,
        eps0=eps0,
        p=p,
        theta=theta,
        n_max=n_max,
        tree=tree,
    )
    return charges[:, None] * field
