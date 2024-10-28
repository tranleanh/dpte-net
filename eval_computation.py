from core.networks import DPTE_Net
from keras_flops import get_flops


if __name__ == '__main__':

    g = DPTE_Net()
    g.summary()

    # MACs
    flops = get_flops(g, batch_size=1)
    print(f"MACs: {flops*0.5 / 10 ** 9:.03} G")       # MACs/FLOPs ~ 1/2
