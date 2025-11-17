def bilayer_flexural_rigidity(h1, h2, Y1, Y2, b1=1.0, b2=1.0):
    """
    Computes the effective flexural rigidity D_eff and the discrete hinge stiffness k_b
    for a bilayer composite.
    """
    # layer centroids (measured from bottom of layer 1)
    y1 = h1 / 2.0
    y2 = h1 + h2 / 2.0

    # neutral axis location
    num = Y1 * b1 * h1 * y1 + Y2 * b2 * h2 * y2
    den = Y1 * b1 * h1     + Y2 * b2 * h2
    ybar = num / den

    # individual layer contributions to D
    D1 = Y1 * b1 * (h1**3 / 12.0 + h1 * (y1 - ybar)**2)
    D2 = Y2 * b2 * (h2**3 / 12.0 + h2 * (y2 - ybar)**2)

    # total effective rigidity and hinge stiffness
    D_eff = D1 + D2

    return D_eff