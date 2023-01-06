DEVICE = 'cpu'
def set_device(value : int) -> None:
    """Either to use cpu or gpu (and which one).
    """
    global DEVICE
    if value >= 0:
        DEVICE = 'cuda:{}'.format(value)
    print(f"DEVICE: {DEVICE}")