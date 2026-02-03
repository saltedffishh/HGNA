# src/UOT.py

import numpy as np
import ot


def compute_cost_matrix(
    mu_t,
    mu_t1,
    prog_t,
    prog_t1,
    alpha=0.5
):
    """
    Compute cost matrix between pseudo-cells.

    mu_t   : (K_t, d1)
    mu_t1  : (K_t1, d1)
    prog_t : (K_t, d2)
    prog_t1: (K_t1, d2)
    """
    # geometric cost
    geo_cost = ot.dist(mu_t, mu_t1, metric="euclidean") ** 2

    # functional cost
    func_cost = ot.dist(prog_t, prog_t1, metric="euclidean") ** 2

    return alpha * geo_cost + (1 - alpha) * func_cost


def unbalanced_ot_align(
    mu_t,
    mu_t1,
    prog_t,
    prog_t1,
    size_t,
    size_t1,
    alpha=0.5,
    reg=0.05,
    reg_m=1.0
):
    """
    Unbalanced Optimal Transport alignment between pseudo-cells.

    Returns:
        pi : transport plan (K_t x K_t1)
    """

    # normalize masses
    a = size_t / size_t.sum()
    b = size_t1 / size_t1.sum()

    # cost matrix
    C = compute_cost_matrix(
        mu_t, mu_t1, prog_t, prog_t1, alpha
    )

    # Unbalanced Sinkhorn
    pi = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C,
        reg=reg,
        reg_m=reg_m
    )

    return pi