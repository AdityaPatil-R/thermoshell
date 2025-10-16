import matplotlib.pyplot as plt

def new_fig(num=None, figsize=(6,6), label_fmt='Fig. {n}',
            label_pos=(0.01, 1.00), **subplot_kw):
    """
    Create (or switch to) figure `num` and a single Axes,
    then write 'Fig. <num>' in the upper‐left corner of the figure.
    Returns (fig, ax).
    
    - num: figure number (int) or None to auto‐increment.
    - figsize: passed through to plt.figure.
    - label_fmt: format string; '{n}' will be replaced by fig.number.
    - label_pos: (x, y) in figure fraction (0–1) for the label.
    - subplot_kw: passed to fig.add_subplot (e.g. projection='3d').
    """
    fig = plt.figure(num, figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1, **subplot_kw)
    
    lbl = label_fmt.format(n=fig.number)
    # place in figure coords, so works for 2D or 3D
    fig.text(
        label_pos[0], label_pos[1], lbl,
        transform=fig.transFigure,
        fontsize=12, fontweight='bold',
        va='top'
    )
    return fig, ax