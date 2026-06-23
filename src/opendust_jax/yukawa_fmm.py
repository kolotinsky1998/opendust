"""Correctness-first screened Coulomb backends built on jaxFMM hierarchies.

This module uses the existing adaptive hierarchy and interaction lists from
jaxFMM, but evaluates a pure Yukawa kernel. The default backend uses analytic
Cartesian Taylor multipoles for well-separated source boxes; a Chebyshev
interpolation backend is kept as a high-accuracy fallback. Neighboring boxes are
computed exactly by direct P2P. The implementation is independent from the
Laplace-specific ``jaxfmm.eval_potential`` path.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from functools import partial
from math import factorial

from opendust_jax.direct import _validate_inputs
from opendust_jax.fmm_poisson import _import_jaxfmm
from opendust_jax.yukawa_basis import (
    modified_spherical_bessel_i,
    modified_spherical_bessel_k,
)


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


def _multi_indices(order: int) -> jax.Array:
    rows = []
    for ax in range(3):
        rows.append((ax,))
    if order == 1:
        return jnp.array(rows, dtype=jnp.int32)
    rows = [(i, j) for i in range(3) for j in range(3)]
    if order == 2:
        return jnp.array(rows, dtype=jnp.int32)
    rows = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    if order == 3:
        return jnp.array(rows, dtype=jnp.int32)
    raise ValueError("Taylor backend currently supports order 0..3.")


def _lm_pairs(order: int) -> tuple[tuple[int, int], ...]:
    return tuple((ell, m) for ell in range(order + 1) for m in range(-ell, ell + 1))


def _associated_legendre(l: int, m_abs: int, x: jax.Array) -> jax.Array:
    pmm = jnp.ones_like(x)
    if m_abs > 0:
        somx2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(1, m_abs + 1):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if l == m_abs:
        return pmm
    pmmp1 = x * (2 * m_abs + 1) * pmm
    if l == m_abs + 1:
        return pmmp1
    pll = pmmp1
    p_lm2 = pmm
    p_lm1 = pmmp1
    for ell in range(m_abs + 2, l + 1):
        pll = ((2 * ell - 1) * x * p_lm1 - (ell + m_abs - 1) * p_lm2) / (ell - m_abs)
        p_lm2 = p_lm1
        p_lm1 = pll
    return pll


def _complex_spherical_harmonic(l: int, m: int, rvec: jax.Array) -> jax.Array:
    r = jnp.linalg.norm(rvec, axis=-1)
    safe_r = jnp.where(r == 0.0, 1.0, r)
    cos_theta = jnp.clip(rvec[..., 2] / safe_r, -1.0, 1.0)
    phi = jnp.arctan2(rvec[..., 1], rvec[..., 0])
    m_abs = abs(m)
    norm = jnp.sqrt(
        (2 * l + 1)
        / (4.0 * jnp.pi)
        * factorial(l - m_abs)
        / factorial(l + m_abs)
    )
    y_pos = norm * _associated_legendre(l, m_abs, cos_theta) * jnp.exp(1j * m_abs * phi)
    if m >= 0:
        y = y_pos
    else:
        y = ((-1) ** m_abs) * jnp.conj(y_pos)
    return jnp.where(r == 0.0, jnp.where(l == 0, 1.0 / jnp.sqrt(4.0 * jnp.pi), 0.0), y)


@partial(jax.jit, static_argnames=("order",))
def _compute_spherical_moments(
    padded_pts: jax.Array,
    padded_chrgs: jax.Array,
    centers: jax.Array,
    kappa: float,
    order: int,
) -> jax.Array:
    offsets = padded_pts - centers[:, None, :]
    rho = jnp.linalg.norm(offsets, axis=-1)
    radial_i = modified_spherical_bessel_i(kappa * rho, order)
    moments = []
    for ell, m in _lm_pairs(order):
        ylm = _complex_spherical_harmonic(ell, m, offsets)
        moments.append(jnp.sum(padded_chrgs * radial_i[..., ell] * jnp.conj(ylm), axis=1))
    return jnp.stack(moments, axis=1)


def _spherical_yukawa_potential_from_moments(
    point: jax.Array,
    center: jax.Array,
    moments: jax.Array,
    kappa: float,
    order: int,
) -> jax.Array:
    rvec = point - center
    r = jnp.linalg.norm(rvec)
    radial_k = modified_spherical_bessel_k(kappa * r, order)
    value = 0.0 + 0.0j
    for idx, (ell, m) in enumerate(_lm_pairs(order)):
        value = value + moments[idx] * radial_k[ell] * _complex_spherical_harmonic(ell, m, rvec)
    # modified_spherical_bessel_k contains the pi/2 convention:
    # k_0(x) = pi/2 * exp(-x) / x. With normalized complex Y_00, the prefactor
    # 8*kappa makes the l=0 source-at-center case exactly exp(-k*r)/r.
    return jnp.real((8.0 * kappa) * value)


def _spherical_yukawa_field_from_moments(
    point: jax.Array,
    center: jax.Array,
    moments: jax.Array,
    kappa: float,
    order: int,
) -> jax.Array:
    grad = jax.grad(_spherical_yukawa_potential_from_moments, argnums=0)(
        point,
        center,
        moments,
        kappa,
        order,
    )
    return -grad


@partial(jax.jit, static_argnames=("order",))
def _eval_spherical_box_field(
    eval_points: jax.Array,
    center: jax.Array,
    moments: jax.Array,
    kappa: float,
    order: int,
) -> jax.Array:
    return jax.vmap(
        lambda point: _spherical_yukawa_field_from_moments(point, center, moments, kappa, order)
    )(eval_points)


@partial(jax.jit, static_argnames=("order",))
def _eval_far_field_spherical(
    padded_eval_pts: jax.Array,
    centers_global: jax.Array,
    moments_global: jax.Array,
    mpl_cnct: jax.Array,
    trg_leaf_offset: int,
    kappa: float,
    order: int,
) -> jax.Array:
    field = jnp.zeros_like(padded_eval_pts)

    def scan_body(buf: jax.Array, pair: jax.Array) -> tuple[jax.Array, None]:
        trg_local = pair[0] - trg_leaf_offset
        src_global = pair[1]
        chunk = _eval_spherical_box_field(
            padded_eval_pts[trg_local],
            centers_global[src_global],
            moments_global[src_global],
            kappa,
            order,
        )
        buf = buf.at[trg_local].add(chunk)
        return buf, None

    field, _ = jax.lax.scan(scan_body, field, mpl_cnct)
    return field


@jax.jit
def _yukawa_field_kernel(rvec: jax.Array, kappa: float) -> jax.Array:
    r = jnp.linalg.norm(rvec)
    inv_r = jnp.where(r == 0.0, 0.0, 1.0 / r)
    screened = jnp.exp(-kappa * r)
    radial = screened * (inv_r**3 + kappa * inv_r**2)
    return radial * rvec


_yukawa_field_jac1 = jax.jit(jax.jacfwd(_yukawa_field_kernel, argnums=0))
_yukawa_field_jac2 = jax.jit(jax.jacfwd(_yukawa_field_jac1, argnums=0))
_yukawa_field_jac3 = jax.jit(jax.jacfwd(_yukawa_field_jac2, argnums=0))


@partial(jax.jit, static_argnames=("order",))
def _compute_taylor_moments(
    padded_pts: jax.Array,
    padded_chrgs: jax.Array,
    centers: jax.Array,
    order: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    offsets = padded_pts - centers[:, None, :]
    m0 = jnp.sum(padded_chrgs, axis=1)
    m1 = -jnp.sum(padded_chrgs[..., None] * offsets, axis=1)
    m2 = jnp.einsum("bn,bni,bnj->bij", padded_chrgs, offsets, offsets) * 0.5
    m3 = -jnp.einsum("bn,bni,bnj,bnk->bijk", padded_chrgs, offsets, offsets, offsets) / 6.0
    if order < 3:
        m3 = jnp.zeros_like(m3)
    if order < 2:
        m2 = jnp.zeros_like(m2)
    if order < 1:
        m1 = jnp.zeros_like(m1)
    return m0, m1, m2, m3


@partial(jax.jit, static_argnames=("order",))
def _eval_taylor_box_field(
    eval_points: jax.Array,
    center: jax.Array,
    m0: jax.Array,
    m1: jax.Array,
    m2: jax.Array,
    m3: jax.Array,
    kappa: float,
    order: int,
) -> jax.Array:
    def eval_one(point: jax.Array) -> jax.Array:
        rvec = point - center
        field = m0 * _yukawa_field_kernel(rvec, kappa)
        if order >= 1:
            d1 = _yukawa_field_jac1(rvec, kappa)
            field = field + jnp.einsum("ci,i->c", d1, m1)
        if order >= 2:
            d2 = _yukawa_field_jac2(rvec, kappa)
            field = field + jnp.einsum("cij,ij->c", d2, m2)
        if order >= 3:
            d3 = _yukawa_field_jac3(rvec, kappa)
            field = field + jnp.einsum("cijk,ijk->c", d3, m3)
        return field

    return jax.vmap(eval_one)(eval_points)


@partial(jax.jit, static_argnames=("order",))
def _eval_far_field_taylor(
    padded_eval_pts: jax.Array,
    centers_global: jax.Array,
    m0_global: jax.Array,
    m1_global: jax.Array,
    m2_global: jax.Array,
    m3_global: jax.Array,
    mpl_cnct: jax.Array,
    trg_leaf_offset: int,
    kappa: float,
    order: int,
) -> jax.Array:
    field = jnp.zeros_like(padded_eval_pts)

    def scan_body(buf: jax.Array, pair: jax.Array) -> tuple[jax.Array, None]:
        trg_local = pair[0] - trg_leaf_offset
        src_global = pair[1]
        chunk = _eval_taylor_box_field(
            padded_eval_pts[trg_local],
            centers_global[src_global],
            m0_global[src_global],
            m1_global[src_global],
            m2_global[src_global],
            m3_global[src_global],
            kappa,
            order,
        )
        buf = buf.at[trg_local].add(chunk)
        return buf, None

    field, _ = jax.lax.scan(scan_body, field, mpl_cnct)
    return field


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


def _yukawa_fmm_field_chebyshev(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    p: int,
    theta: float,
    n_max: int,
    tree: dict[str, Any],
    eps0: float,
) -> jax.Array:
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


def _yukawa_fmm_field_taylor(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    p: int,
    theta: float,
    n_max: int,
    tree: dict[str, Any],
    eps0: float,
) -> jax.Array:
    padded = handle_padding(tree["pts"], charges, tree["eval_pts"], tree["idcs"])
    padded_pts, padded_chrgs, padded_eval_pts = padded[:3]
    dir_padded_pts, dir_padded_chrgs, dir_padded_eval_pts = padded[3:]

    order = min(int(p), 3)
    src_lvl = tree["lvl_info"][-2][1]
    trg_lvl = tree["lvl_info"][-2][0]
    src_leaf_offset = tree["src_ofs"][src_lvl]
    trg_leaf_offset = tree["trg_ofs"][trg_lvl]
    src_next_offset = tree["src_ofs"][src_lvl + 1]

    source_centers = tree["boxcenters"][src_leaf_offset:src_next_offset]
    m0, m1, m2, m3 = _compute_taylor_moments(
        padded_pts,
        padded_chrgs,
        source_centers,
        order,
    )

    n_global_src = tree["boxcenters"].shape[0]
    centers_global = tree["boxcenters"]
    m0_global = jnp.zeros((n_global_src,), dtype=charges.dtype)
    m1_global = jnp.zeros((n_global_src, 3), dtype=charges.dtype)
    m2_global = jnp.zeros((n_global_src, 3, 3), dtype=charges.dtype)
    m3_global = jnp.zeros((n_global_src, 3, 3, 3), dtype=charges.dtype)
    m0_global = m0_global.at[src_leaf_offset:src_next_offset].set(m0)
    m1_global = m1_global.at[src_leaf_offset:src_next_offset].set(m1)
    m2_global = m2_global.at[src_leaf_offset:src_next_offset].set(m2)
    m3_global = m3_global.at[src_leaf_offset:src_next_offset].set(m3)

    if tree["mpl_cnct"].size == 0:
        far_leaf = jnp.zeros_like(padded_eval_pts)
    else:
        far_leaf = _eval_far_field_taylor(
            padded_eval_pts,
            centers_global,
            m0_global,
            m1_global,
            m2_global,
            m3_global,
            tree["mpl_cnct"],
            trg_leaf_offset,
            kappa,
            order,
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


def _yukawa_fmm_field_spherical(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    p: int,
    theta: float,
    n_max: int,
    tree: dict[str, Any],
    eps0: float,
) -> jax.Array:
    padded = handle_padding(tree["pts"], charges, tree["eval_pts"], tree["idcs"])
    padded_pts, padded_chrgs, padded_eval_pts = padded[:3]
    dir_padded_pts, dir_padded_chrgs, dir_padded_eval_pts = padded[3:]

    order = int(p)
    src_lvl = tree["lvl_info"][-2][1]
    trg_lvl = tree["lvl_info"][-2][0]
    src_leaf_offset = tree["src_ofs"][src_lvl]
    trg_leaf_offset = tree["trg_ofs"][trg_lvl]
    src_next_offset = tree["src_ofs"][src_lvl + 1]

    source_centers = tree["boxcenters"][src_leaf_offset:src_next_offset]
    moments = _compute_spherical_moments(
        padded_pts,
        padded_chrgs,
        source_centers,
        kappa,
        order,
    )

    n_global_src = tree["boxcenters"].shape[0]
    n_coeff = (order + 1) ** 2
    moments_global = jnp.zeros((n_global_src, n_coeff), dtype=jnp.complex64)
    moments_global = moments_global.at[src_leaf_offset:src_next_offset].set(moments)

    if tree["mpl_cnct"].size == 0:
        far_leaf = jnp.zeros_like(padded_eval_pts)
    else:
        far_leaf = _eval_far_field_spherical(
            padded_eval_pts,
            tree["boxcenters"],
            moments_global,
            tree["mpl_cnct"],
            trg_leaf_offset,
            kappa,
            order,
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


def yukawa_fmm_field(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    eps0: float = 8.85418781762039e-12,
    p: int = 4,
    theta: float = 0.77,
    n_max: int = 128,
    tree: dict[str, Any] | None = None,
    backend: str = "taylor",
) -> jax.Array:
    """Evaluate the free-space Yukawa electric field using a screened backend."""

    positions, charges = _validate_inputs(positions, charges)
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")
    if eps0 <= 0:
        raise ValueError("eps0 must be positive.")
    if tree is None:
        tree = build_yukawa_tree(positions, n_max=n_max, theta=theta, p=p)

    _import_jaxfmm()
    from jaxfmm.hierarchy import handle_padding as _handle_padding

    globals()["handle_padding"] = _handle_padding
    if backend == "taylor":
        return _yukawa_fmm_field_taylor(
            positions,
            charges,
            kappa,
            p,
            theta,
            n_max,
            tree,
            eps0,
        )
    if backend == "spherical":
        return _yukawa_fmm_field_spherical(
            positions,
            charges,
            kappa,
            p,
            theta,
            n_max,
            tree,
            eps0,
        )
    if backend == "chebyshev":
        return _yukawa_fmm_field_chebyshev(
            positions,
            charges,
            kappa,
            p,
            theta,
            n_max,
            tree,
            eps0,
        )
    raise ValueError("backend must be 'spherical', 'taylor', or 'chebyshev'.")


def yukawa_fmm_forces(
    positions: jax.Array,
    charges: jax.Array,
    kappa: float,
    eps0: float = 8.85418781762039e-12,
    p: int = 4,
    theta: float = 0.77,
    n_max: int = 128,
    tree: dict[str, Any] | None = None,
    backend: str = "taylor",
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
        backend=backend,
    )
    return charges[:, None] * field
