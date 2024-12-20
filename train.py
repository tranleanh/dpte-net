import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tqdm import tqdm
import numpy as np

from core.utils import load_hazy_clean_hint_soft_trans
from core.losses import wasserstein_loss, perceptual_and_l2_loss, l2_loss, l2_loss_hint, \
                            perceptual_and_transmission_guided_loss, \
                            soft_perceptual_and_transmission_guided_loss, soft_l2_loss_hint

from core.networks import DPTE_Net, Critic
from core.networks import img_size

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam


BASE_DIR = 'weights/'


d_weight_path = ''
g_weight_path = ''


def save_all_weights(d, g, epoch_number, current_loss):
    save_dir_g = os.path.join(BASE_DIR, 'g')
    if not os.path.exists(save_dir_g):
        os.makedirs(save_dir_g)

    save_dir_d = os.path.join(BASE_DIR, 'd')
    if not os.path.exists(save_dir_d):
        os.makedirs(save_dir_d)

    g.save_weights(os.path.join(save_dir_g, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir_d, 'discriminator_{}.h5'.format(epoch_number)), True)


def train(n_images, batch_size, log_dir, epoch_num, critic_updates=5):

    data = load_hazy_clean_hint_soft_trans('path/to/train/data', n_images)
    x_train, y_train, h_train, s_train, g_train = data['A'], data['B'], data['C'], data['D'], data['E']

    slabel_decays = []
    for i in range(epoch_num):
        rate = [(epoch_num - i)/epoch_num]
        rate = np.array(rate)
        rate = np.reshape(rate, (1,1,1))
        slabel_decays.append(rate)

    print("Total data:", len(y_train))


    # Define GAN Model
    g = DPTE_Net()
    d = Critic()

    inputs = Input(shape=(img_size,img_size,4))
    guides = Input(shape=(img_size,img_size,1))
    decays = Input(shape=(1,1))

    generated_image, guided_fm, generated_image_2 = g(inputs)
    outputs = d(generated_image)

    d_on_g = Model(inputs=[inputs, guides, decays], outputs=[generated_image, guided_fm, generated_image_2, outputs])


    # Load pre-trained weights
    if g_weight_path != "" and d_weight_path != "":
        g.load_weights(g_weight_path)
        d.load_weights(d_weight_path)

    lr = 1E-4
    decay_rate = lr/epoch_num*0.5

    d_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
    d_on_g_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)

    # d_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # d_on_g_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False


    # Define Loss Function
    loss = [perceptual_and_transmission_guided_loss(guide=guides), \
            soft_l2_loss_hint(slabel_decay=decays), \
            soft_perceptual_and_transmission_guided_loss(guide=guides, slabel_decay=decays), wasserstein_loss]

    loss_weights = [30, 10, 10, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    for e, epoch in enumerate(range(epoch_num)):

        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []


        print('Epoch ' + str(e+1) + ' / ' + str(epoch_num))
        # print(loss_weights)

        slabel_decay = slabel_decays[epoch]

        for index in tqdm(range(int(x_train.shape[0] / batch_size))):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]

            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]
            image_hint_batch = h_train[batch_indexes]
            image_soft_batch = s_train[batch_indexes]
            guide_tran_batch = g_train[batch_indexes]

            generated_images, _, __ = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch([image_blur_batch, guide_tran_batch, slabel_decay], [image_full_batch, image_hint_batch, image_soft_batch, output_true_batch])
            # print(d_on_g_loss)
            d_on_g_losses.append(d_on_g_loss)

            d.trainable = True

        print("-> DLoss:", np.mean(d_losses), "- GLoss", np.mean(d_on_g_losses))

        epoch_ = epoch+1
        if epoch_ % 5 == 0:
            save_all_weights(d, g, epoch_, int(np.mean(d_on_g_losses)))


if __name__ == '__main__':

    # Train Parameters:
    n_images = 4
    batch_size = 2
    log_dir = False
    epoch_num = 200
    critic_updates = 5

    # Train Network
    train(n_images, batch_size, log_dir, epoch_num, critic_updates)


