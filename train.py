import os
import time
import traceback
from multiprocessing.spawn import freeze_support

import numpy as np
import torch
from termcolor import colored

# from infer_video_hd_1024 import IMG_HEIGHT
from sl_dataset import ImagePathDataset_sl
# from dataset_test import MyMapStyleDataset

from GAN2_hd_split_xl_cd7_RTM_attention_adain_X2_natn_head_new import Generator, Discriminator, Discriminator_Hand, \
    Discriminator_Head

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchinfo import summary

import torch.nn.functional as F
from torchvision.utils import save_image
from torch.distributions import uniform
from adabelief_pytorch import AdaBelief

from torchvision.transforms import Resize, InterpolationMode

print(torch.__version__)
IMG_WIDTH = 512
IMG_HEIGHT = 512

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# TRAIN_STEP = 4000000

num_dtype = torch.float32

if not os.path.exists(f'training_checkpoints_split_raw_cd'):
    os.makedirs(f'training_checkpoints_split_raw_cd')
if not os.path.exists(f'images_split_raw_cd'):
    os.makedirs(f'images_split_raw_cd')
checkpoint_dir = './training_checkpoints_split_raw_cd'

loss_object = torch.nn.BCEWithLogitsLoss()
loss_fn_s = torch.nn.MSELoss()

LAMBDA_hand = 1
LAMBDA_head = 1


def resize_with_crop_or_pad_torch(tensor, target_height, target_width):
    _, H, W = tensor.shape

    # crop
    if H > target_height:
        top = (H - target_height) // 2
        tensor = tensor[:, top: top + target_height, :]
        _, H, W = tensor.shape

    if W > target_width:
        left = (W - target_width) // 2
        tensor = tensor[:, :, left: left + target_width]
        _, H, W = tensor.shape

    # 居中
    pad_left = pad_right = pad_top = pad_bottom = 0

    if H < target_height:
        pad_top = (target_height - H) // 2
        pad_bottom = target_height - H - pad_top

    if W < target_width:
        pad_left = (target_width - W) // 2
        pad_right = target_width - W - pad_left

    # padding
    out = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    return out


def generator_loss(disc_generated_output, disc_generated_output_hand, disc_generated_output_head, need_train_D_hand,
                   gen_output, target, LAMBDA,
                   num_dtype=torch.float32):
    # global LAMBDA
    distribution = uniform.Uniform(torch.Tensor([0.95]), torch.Tensor([1.0]))
    # print(disc_generated_output.shape)
    rd_num_d = distribution.sample(disc_generated_output.shape).squeeze(-1).cuda()
    # print(rd_num_d.shape)
    gan_loss_all = loss_object(disc_generated_output,
                               rd_num_d)
    if need_train_D_hand:
        rd_num_d_hand = distribution.sample(disc_generated_output_hand.shape).squeeze(-1).cuda()

        gan_loss_hand = loss_object(disc_generated_output_hand,
                                    rd_num_d_hand)
    else:
        gan_loss_hand = 0.

    rd_num_d_head = distribution.sample(disc_generated_output_head.shape).squeeze(-1).cuda()

    gan_loss_head = loss_object(disc_generated_output_head,
                                rd_num_d_head)
    # Mean absolute error
    l1_loss = torch.mean(torch.abs(target - gen_output), dtype=num_dtype)

    total_gen_loss = gan_loss_all + (LAMBDA_hand * gan_loss_hand) + (LAMBDA_head * gan_loss_head) + (
            LAMBDA * l1_loss)  # + loss_style

    return total_gen_loss, gan_loss_all, gan_loss_hand, gan_loss_head, l1_loss


# disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()
# plt.show()

def discriminator_loss(disc_real_output, disc_generated_output, num_dtype=torch.float32):
    # need_reverse = torch.FloatTensor(0.0, 1.).uniform_(1).type(num_dtype)
    distribution_r = uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([1.0]))
    distribution_1 = uniform.Uniform(torch.Tensor([0.95]), torch.Tensor([1.0]))
    distribution_0 = uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([0.05]))

    need_reverse = distribution_r.sample([1]).squeeze().cuda()
    # print(need_reverse)
    if need_reverse > 0.01:
        # print("normal")
        rd_num_r = distribution_1.sample(disc_generated_output.shape).squeeze(-1).cuda()
        real_loss = loss_object(disc_real_output,
                                rd_num_r)
        rd_num_g = distribution_0.sample(disc_generated_output.shape).squeeze(-1).cuda()
        generated_loss = loss_object(disc_generated_output,
                                     rd_num_g)
    else:
        # print("reverse")
        rd_num_r = distribution_0.sample(disc_generated_output.shape).squeeze(-1).cuda()
        real_loss = loss_object(disc_real_output,
                                rd_num_r)
        rd_num_g = distribution_1.sample(disc_generated_output.shape).squeeze(-1).cuda()
        generated_loss = loss_object(disc_generated_output,
                                     rd_num_g)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
generator = Generator()
generator = generator.cuda()
total_params = sum(p.numel() for p in generator.parameters())
print(f"generator Number of parameters: {total_params}")

discriminator = Discriminator()
discriminator = discriminator.cuda()
total_params = sum(p.numel() for p in discriminator.parameters())
print(f"discriminator Number of parameters: {total_params}")

discriminator_hand = Discriminator_Hand()
discriminator_hand = discriminator_hand.cuda()
total_params = sum(p.numel() for p in discriminator_hand.parameters())
print(f"discriminator_hand Number of parameters: {total_params}")

discriminator_head = Discriminator_Head()
discriminator_head = discriminator_head.cuda()
total_params = sum(p.numel() for p in discriminator_head.parameters())
print(f"discriminator_head Number of parameters: {total_params}")

# checkpoint_dir = './training_checkpoints_split_raw_cd'
# g_checkpoint_pt = '/model_g_20250117_174458_400000.pt'
# g_checkpoint_path = checkpoint_dir + g_checkpoint_pt
# generator.load_state_dict(torch.load(g_checkpoint_path, weights_only=True))
# d_checkpoint_pt = '/model_d_20250117_174458_400000.pt'
# d_checkpoint_path = checkpoint_dir + d_checkpoint_pt
# discriminator.load_state_dict(torch.load(d_checkpoint_path, weights_only=True))
# dh_checkpoint_pt = '/model_dh_20250117_174458_400000.pt'
# dh_checkpoint_path = checkpoint_dir + dh_checkpoint_pt
# discriminator_hand.load_state_dict(torch.load(dh_checkpoint_path, weights_only=True))

# summary(generator, torch.zeros([1, 4, 512, 512]).to(device),torch.zeros([1, 3, 512, 512]).to(device),torch.zeros([1, 23, 2]).to(device),torch.zeros([1, 68, 2]).to(device),torch.zeros([1, 21, 2]).to(device),torch.zeros([1, 21, 2]).to(device), show_input=True, print_summary=True, show_hierarchical=True)
# summary(discriminator, torch.zeros([1, 7, 512, 512]).to(device),torch.zeros([1, 3, 512, 512]).to(device), show_input=True, print_summary=True)
# summary(discriminator_hand, torch.zeros([1, 3, 512, 512]).to(device), show_input=True, print_summary=True)
summary(generator, [[1, 3, 512, 512], [1, 3, 512, 512], [1, 23, 2], [1, 68, 2], [1, 21, 2], [1, 21, 2]],
        dtypes=[torch.float, torch.float, torch.float, torch.float, torch.float, torch.float])
summary(discriminator, [[1, 3, 512, 512], [1, 3, 512, 512]], dtypes=[torch.float, torch.float])
summary(discriminator_hand, [[1, 3, 128, 128],[1, 3, 128, 128]], dtypes=[torch.float,torch.float])
summary(discriminator_head, [[1, 3, 128, 128],[1, 3, 128, 128]], dtypes=[torch.float,torch.float])

generator_optimizer = AdaBelief(generator.parameters(), lr=2e-6, betas=(0.5, 0.999))
generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=0.97)

discriminator_optimizer = AdaBelief(discriminator.parameters(), lr=2e-6, betas=(0.5, 0.999))
discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=0.97)

discriminator_hand_optimizer = AdaBelief(generator.parameters(), lr=2e-6, betas=(0.5, 0.999))
discriminator_hand_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=discriminator_hand_optimizer,
                                                                      gamma=0.97)
discriminator_head_optimizer = AdaBelief(generator.parameters(), lr=2e-6, betas=(0.5, 0.999))
discriminator_head_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=discriminator_head_optimizer,
                                                                      gamma=0.97)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

torch_resize = Resize([128, 128], interpolation=InterpolationMode.BICUBIC)  # 定义Resize类对象


def generate_images(model, example_input_image, example_style_image, npy_list, example_real_image):
    npy_list = torch.nan_to_num(npy_list)
    bodies = npy_list[:, 0:23, :]
    faces = npy_list[:, 23:91, :]
    left_hands = npy_list[:, 91:112, :]
    right_hands = npy_list[:, 112:, :]
    content = example_input_image
    style = example_style_image
    prediction = model(content, style, bodies, faces, left_hands, right_hands)
    # tset_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer("add_7").output)
    # prd_test = tset_layer.predict(test_input)
    # print(prd_test)
    # y, idx = tf.unique(tf.math.is_nan(tf.reshape(prd_test, [-1])))
    # print(y)
    return example_real_image, prediction


def resize_cut_image(gen_img_cut_l, gen_img_cut_r):
    # print("resize_cut_image tracing")
    gen_img_cut_l = resize_with_crop_or_pad_torch(gen_img_cut_l, 128, 128)
    gen_img_cut_r = resize_with_crop_or_pad_torch(gen_img_cut_r, 128, 128)
    # print(gen_img_cut_l.shape)
    gen_output_cut_list = torch.stack([gen_img_cut_l, gen_img_cut_r], dim=0)
    return gen_output_cut_list, gen_img_cut_l, gen_img_cut_r


def resize_cut_image_head(gen_img_cut_head):
    # print("resize_cut_image tracing")
    gen_img_cut_head = torch_resize(gen_img_cut_head)

    # print(gen_img_cut_l.shape)
    gen_output_cut_head_list = torch.stack([gen_img_cut_head], dim=0)
    return gen_output_cut_head_list, gen_img_cut_head


def train_step(content, style, hand_img_f,hand_input_img_f, bund_list, head_img_list,head_input_img_list, head_bund_list, npy_list, target, LAMBDA,
               step_now):
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    discriminator_hand_optimizer.zero_grad()
    discriminator_head_optimizer.zero_grad()

    # loss_style = 0.0
    npy_list = torch.nan_to_num(npy_list)
    # print("train_step tracing")
    LAMBDA = np.array(LAMBDA)
    LAMBDA = torch.from_numpy(LAMBDA).type(num_dtype)
    LAMBDA.cuda()
    bodies = npy_list[:, 0:23, :].cuda()
    faces = npy_list[:, 23:91, :].cuda()
    left_hands = npy_list[:, 91:112, :].cuda()
    right_hands = npy_list[:, 112:, :].cuda()
    # print(content.shape)
    content = content.cuda()
    style = style.cuda()
    target = target.cuda()
    hand_img_f = hand_img_f.cuda()
    hand_input_img_f = hand_input_img_f.cuda()
    head_img_list = head_img_list.cuda()
    head_input_img_list = head_input_img_list.cuda()

    # input_image = torch.concat([content, style], dim=1)
    # print(content)
    gen_output = generator(content, style, bodies, faces, left_hands, right_hands)
    # print(gen_output)
    # input_image = input_image[:, :, :, 0:4]
    disc_real_output = discriminator(content, target)
    disc_generated_output = discriminator(content, gen_output)
    # with tf.GradientTape() as gen_tape:
    #     gen_output = generator(input_image, training=True)
    #     with tf.GradientTape() as disc_tape:
    #         disc_loss, disc_generated_output = tape_of__discriminator(input_image, target, gen_output)

    # gen_output_cut_list = []
    # gen_output_cut_list_l = []
    # gen_output_cut_list_r = []
    # index = 0
    # for idx, gen_img in enumerate(gen_output):

    # # print(gen_output_cut_list.shape)

    gen_img = gen_output[0]
    # print(gen_img.shape)
    bund_list_item = bund_list[0]
    head_bund_list_item = head_bund_list[0]

    gen_img_cut_l = gen_img[:, bund_list_item[0]:bund_list_item[1], bund_list_item[2]:bund_list_item[3]]
    gen_img_cut_r = gen_img[:, bund_list_item[4]:bund_list_item[5], bund_list_item[6]:bund_list_item[7]]
    gen_img_cut_head = gen_img[:, head_bund_list_item[0]:head_bund_list_item[1],
                       head_bund_list_item[2]:head_bund_list_item[3]]

    gen_output_cut_list, gen_img_cut_l, gen_img_cut_r = resize_cut_image(gen_img_cut_l, gen_img_cut_r)
    gen_output_cut_head_list, gen_img_cut_head = resize_cut_image_head(gen_img_cut_head)

    # gen_output_cut_list, gen_img_cut_l, gen_img_cut_r = cut_image(gen_img, bund_list_item)
    # hand_input_f
    disc_real_output_hand = discriminator_hand(hand_input_img_f,hand_img_f)
    disc_generated_output_hand = discriminator_hand(hand_input_img_f,gen_output_cut_list)

    disc_real_output_head = discriminator_head(head_input_img_list,head_img_list)
    disc_generated_output_head = discriminator_head(head_input_img_list,gen_output_cut_head_list)

    # with tf.GradientTape() as disc_hand_tape:
    #     disc_loss_hand, disc_generated_output_hand = tape_of__discriminator_hand(hand_img_f, gen_output_cut_list)

    # reconstructed_vgg_features = loss_net(gen_output)
    # style_vgg_features = loss_net(style)
    # for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
    #     mean_inp, std_inp = get_mean_std(inp)
    #     mean_out, std_out = get_mean_std(out)
    #     loss_style += loss_fn_s(mean_inp, mean_out) + loss_fn_s(
    #         std_inp, std_out
    #     )
    # loss_style = 0
    # loss_style = 0.1 * loss_style
    gen_total_loss, gen_gan_loss_all, gen_gan_loss_hand, gen_gan_loss_head, gen_l1_loss = generator_loss(
        disc_generated_output,
        disc_generated_output_hand, disc_generated_output_head, True,
        gen_output, target, LAMBDA)
    #
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    # print(disc_real_output)
    # print(disc_generated_output)
    # print(disc_loss)
    disc_loss_hand = discriminator_loss(disc_real_output_hand, disc_generated_output_hand)

    disc_loss_head = discriminator_loss(disc_real_output_head, disc_generated_output_head)

    gen_total_loss.backward(retain_graph=True)
    disc_loss.backward(retain_graph=True)
    disc_loss_hand.backward(retain_graph=True)
    disc_loss_head.backward()

    generator_optimizer.step()
    discriminator_optimizer.step()
    discriminator_hand_optimizer.step()
    discriminator_head_optimizer.step()

    if (step_now + 1) % 80000 == 0:
        generator_scheduler.step()
        discriminator_scheduler.step()
        discriminator_hand_scheduler.step()
        discriminator_head_scheduler.step()

    glr = generator_scheduler.get_last_lr()[0]
    dlr = discriminator_scheduler.get_last_lr()[0]
    dhlr = discriminator_hand_scheduler.get_last_lr()[0]
    dhelr = discriminator_head_scheduler.get_last_lr()[0]

    # print(glr[0])
    gen_total_loss = gen_total_loss.detach().cpu().numpy()
    gen_gan_loss_all = gen_gan_loss_all.detach().cpu().numpy()
    gen_gan_loss_hand = gen_gan_loss_hand.detach().cpu().numpy()
    gen_gan_loss_head = gen_gan_loss_head.detach().cpu().numpy()


    gen_l1_loss = gen_l1_loss.detach().cpu().numpy()
    disc_loss_hand = disc_loss_hand.detach().cpu().numpy()
    disc_loss_head = disc_loss_head.detach().cpu().numpy()

    # loss_style=loss_style.detach().cpu().numpy()
    disc_loss = disc_loss.detach().cpu().numpy()
    LAMBDA = LAMBDA.detach().cpu().numpy()

    # gen_img_cut_l=disc_loss_hand.detach().cpu().numpy()
    # gen_img_cut_r=gen_total_loss.detach().cpu().numpy()
    gen_img_cut_l, gen_img_cut_r, gen_img, gen_img_cut_head = gen_img_cut_l.detach().cpu(), gen_img_cut_r.detach().cpu(), gen_img.detach().cpu(), gen_img_cut_head.detach().cpu()

    return gen_total_loss, gen_gan_loss_all, gen_gan_loss_hand,gen_gan_loss_head, gen_l1_loss, disc_loss, disc_loss_hand, disc_loss_head, glr, dlr, dhlr, dhelr, gen_img_cut_l, gen_img_cut_r, gen_img_cut_head, gen_img, LAMBDA


def train_step_nohand(content, style, target, head_img_list,head_input_img_list, head_bund_list, npy_list, LAMBDA, step_now):
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    discriminator_hand_optimizer.zero_grad()
    discriminator_head_optimizer.zero_grad()

    loss_style = 0.0
    npy_list = torch.nan_to_num(npy_list)
    # print("train_step_nohand tracing")
    LAMBDA = np.array(LAMBDA)
    LAMBDA = torch.from_numpy(LAMBDA).type(num_dtype)
    bodies = npy_list[:, 0:23, :].cuda()
    faces = npy_list[:, 23:91, :].cuda()
    left_hands = npy_list[:, 91:112, :].cuda()
    right_hands = npy_list[:, 112:, :].cuda()
    content = content.cuda()
    style = style.cuda()
    target = target.cuda()
    head_img_list = head_img_list.cuda()
    head_input_img_list = head_input_img_list.cuda()


    # input_image = torch.concat([content, style], dim=1)
    gen_output = generator(content, style, bodies, faces, left_hands, right_hands)

    gen_img = gen_output[0]
    head_bund_list_item = head_bund_list[0]
    gen_img_cut_head = gen_img[:, head_bund_list_item[0]:head_bund_list_item[1],
                       head_bund_list_item[2]:head_bund_list_item[3]]

    gen_output_cut_head_list, gen_img_cut_head = resize_cut_image_head(gen_img_cut_head)

    # input_image = input_image[:, :, :, 0:4]
    disc_real_output = discriminator(content, target)
    disc_generated_output = discriminator(content, gen_output)

    disc_real_output_head = discriminator_head(head_input_img_list,head_img_list)
    disc_generated_output_head = discriminator_head(head_input_img_list,gen_output_cut_head_list)
    # with tf.GradientTape() as gen_tape:
    #     gen_output = generator(input_image, training=True)
    #     with tf.GradientTape() as disc_tape:
    #         disc_loss, disc_generated_output = tape_of__discriminator(input_image, target, gen_output)

    # gen_output_cut_list = []
    # gen_output_cut_list_l = []
    # gen_output_cut_list_r = []
    # index = 0
    # for idx, gen_img in enumerate(gen_output):

    # # print(gen_output_cut_list.shape)

    # with tf.GradientTape() as disc_hand_tape:
    #     disc_loss_hand, disc_generated_output_hand = tape_of__discriminator_hand(hand_img_f, gen_output_cut_list)

    # reconstructed_vgg_features = loss_net(gen_output)
    # style_vgg_features = loss_net(style)
    # for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
    #     mean_inp, std_inp = get_mean_std(inp)
    #     mean_out, std_out = get_mean_std(out)
    #     loss_style += loss_fn_s(mean_inp, mean_out) + loss_fn_s(
    #         std_inp, std_out
    #     )
    # loss_style = 0
    loss_style = 0.1 * loss_style
    gen_total_loss, gen_gan_loss_all, gen_gan_loss_hand,gen_gan_loss_head, gen_l1_loss = generator_loss(
        disc_generated_output,
        0.0,disc_generated_output_head,
        False,
        gen_output, target, LAMBDA)
    #
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    disc_loss_head = discriminator_loss(disc_real_output_head, disc_generated_output_head)

    gen_total_loss.backward(retain_graph=True)
    disc_loss.backward(retain_graph=True)
    disc_loss_head.backward()

    generator_optimizer.step()
    discriminator_optimizer.step()
    discriminator_head_optimizer.step()

    if (step_now + 1) % 80000 == 0:
        generator_scheduler.step()
        discriminator_scheduler.step()
        discriminator_hand_scheduler.step()
        discriminator_head_scheduler.step()

    glr = generator_scheduler.get_last_lr()[0]
    dlr = discriminator_scheduler.get_last_lr()[0]
    dhlr = discriminator_hand_scheduler.get_last_lr()[0]
    dhelr = discriminator_head_scheduler.get_last_lr()[0]

    gen_total_loss = gen_total_loss.detach().cpu().numpy()
    gen_gan_loss_all = gen_gan_loss_all.detach().cpu().numpy()
    # gen_gan_loss_hand = gen_gan_loss_hand.detach().cpu().numpy()
    gen_gan_loss_head=gen_gan_loss_head.detach().cpu().numpy()

    # gen_gan_loss_hand = gen_gan_loss_hand.detach().cpu().numpy()
    gen_l1_loss = gen_l1_loss.detach().cpu().numpy()
    # loss_style=loss_style.detach().cpu().numpy()
    disc_loss = disc_loss.detach().cpu().numpy()
    disc_loss_head = disc_loss_head.detach().cpu().numpy()

    LAMBDA = LAMBDA.detach().cpu().numpy()
    disc_loss_hand = 0.0

    gen_img_cut_head = gen_img_cut_head.detach().cpu()
    gen_img = gen_img.detach().cpu()

    return gen_total_loss, gen_gan_loss_all, gen_gan_loss_hand,gen_gan_loss_head, gen_l1_loss, disc_loss, disc_loss_hand, disc_loss_head, glr, dlr, dhlr, dhelr, gen_img_cut_head, gen_img, LAMBDA


def train_loop():
    training_set = ImagePathDataset_sl(r"RTM_train_split_raw_all")
    validation_set = ImagePathDataset_sl(r"RTM_val_split_raw_all")
    # training_set_i=LimitedStepsCycleDataset(training_set,steps=TRAIN_STEP)
    # validation_set_i = LimitedStepsCycleDataset(validation_set, steps=TRAIN_STEP)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, num_workers=0, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=2, shuffle=True)
    training_set_len = len(training_loader)
    validation_set_len = len(validation_loader)
    # train_loader_iter = iter(training_loader)
    val_loader_iter = iter(validation_loader)

    print('Training set has {} instances'.format(training_set_len))
    print('Validation set has {} instances'.format(validation_set_len))
    print(training_loader)
    # inf_train_loader = iter(training_loader)
    # inf_val_loader = iter(validation_loader)
    step_now = 0
    step_stop = 20000000
    step_init = 0
    generator.train(True)
    discriminator.train(True)
    discriminator_hand.train(True)
    val_step = 0
    val_step_gap = 500
    start_10 = time.time()
    start = time.time()
    while step_now <= step_stop:
        if step_now % training_set_len == 0:
            train_loader_iter = iter(training_loader)
        # print("into loop")
        # for i_batch, (input_image, target, style_image, img_path) in enumerate(training_loader):
        try:
            step_real = step_init + step_now
            # if 100000 >= step_real > 0:
            #     LAMBDA = 5 * (1 - step_real / 100000)
            # elif step_real > 100000:
            #     LAMBDA = 0.0
            # else:
            #     LAMBDA = 5.0
            #
            # if step_init>100000 and step_now<=2000:
            #     LAMBDA = 5.0
            LAMBDA = 100.0
            # print(start)

            # print(img_path.shape)

            # print("step")
            # if step_now % 10:
            #     print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')
            if step_now % val_step_gap == 0:
                # display.clear_output(wait=True)
                generator_optimizer.zero_grad()
                generator.eval()

                with torch.no_grad():
                    if step_now != 0:
                        print(f'Time taken for {val_step_gap} steps: {time.time() - start:.2f} sec\n')

                    start = time.time()
                    example_input_image, example_real_image, example_style_image, example_image_file = next(
                        val_loader_iter)  # inf_val_loader.next()
                    image_file_str_group = example_image_file
                    # print(image_file_str_group)
                    hand_img_list = []
                    # hand_R_img_list = []
                    bund_list = []
                    npy_list = []
                    for image_file_str in image_file_str_group:
                        encoding = 'utf-8'
                        # image_file_str = str(image_file_str, encoding)  # print(image_file_str)
                        image_file_str_nodot = image_file_str.replace('\\', '/').split(".")[0]
                        npflie = np.load("./" + image_file_str_nodot + ".npy")
                        # print(npflie)
                        npy_list.append(npflie)
                    npy_list_np = np.array(npy_list)
                    npy_list_ts = torch.from_numpy(npy_list_np).type(num_dtype)
                    # print(npy_list.shape)

                    input_body_img = example_input_image[:, 0:1, :, :]
                    input_left_hand_img = example_input_image[:, 1:2, :, :]
                    input_right_hand_img = example_input_image[:, 2:3, :, :]
                    input_face_img = example_input_image[:, 3:4, :, :]
                    # print(torch.max(input_body_img))
                    # print(torch.max(input_face_img))
                    input_body_and_fance_img = ((input_body_img + 1) + (input_face_img + 1)) - 1
                    input_body_and_fance_img = torch.clip(input_body_and_fance_img, -1., 1.)
                    # print(torch.max(input_body_and_fance_img))
                    processed_input_img = torch.concat(
                        [input_body_and_fance_img, input_left_hand_img, input_right_hand_img], dim=1)

                    tar, predict = generate_images(generator, processed_input_img.to(device),
                                                   example_style_image.to(device), npy_list_ts.to(device),
                                                   example_real_image.to(device))
                    tar, predict = tar.cpu(), predict.cpu()
                    # print(predict)
                    index = 0
                    head_bund_list = []
                    head_img_list = []
                    for image_file_str in image_file_str_group:
                        encoding = 'utf-8'
                        # image_file_str = str(image_file_str, encoding)
                        # print(image_file_str)

                        nose_point = (npy_list[index][53] * IMG_HEIGHT).astype(np.int32)
                        hand_img_b = torch.ones([3, IMG_HEIGHT, IMG_WIDTH])

                        # print(nose_point)
                        image_file_str_L_X_N = nose_point[0] - 96
                        image_file_str_L_Y_N = nose_point[1] - 96
                        image_file_str_R_X_N = nose_point[0] + 96
                        image_file_str_R_Y_N = nose_point[1] + 96

                        X_cut_L_N = image_file_str_L_X_N
                        Y_cut_L_N = image_file_str_L_Y_N

                        X_cut_R_N = image_file_str_R_X_N
                        Y_cut_R_N = image_file_str_R_Y_N

                        tar_numpy = tar[index]
                        tar_numpy = torch.permute(tar_numpy, (1, 2, 0))
                        predict_numpy = predict[index]
                        predict_numpy = torch.permute(predict_numpy, (1, 2, 0))

                        # head_img = np.ones([IMG_HEIGHT, IMG_WIDTH, 3])

                        head_img_cut = tar_numpy[Y_cut_L_N:Y_cut_R_N, X_cut_L_N:X_cut_R_N, :]

                        head_img_cut = torch.permute(head_img_cut, [2, 0, 1])
                        head_img_cut = torch_resize(head_img_cut)

                        head_img_cut_p = predict_numpy[Y_cut_L_N:Y_cut_R_N, X_cut_L_N:X_cut_R_N, :]

                        head_img_cut_p = torch.permute(head_img_cut_p, [2, 0, 1])
                        head_img_cut_p = torch_resize(head_img_cut_p)

                        hand_img_b[:, IMG_WIDTH // 3:IMG_WIDTH // 3 + head_img_cut.shape[1],
                        0:head_img_cut.shape[2]] = head_img_cut
                        hand_img_b[:, IMG_WIDTH // 3:IMG_WIDTH // 3 + head_img_cut_p.shape[1],
                        IMG_WIDTH - head_img_cut_p.shape[2]:IMG_WIDTH] = head_img_cut_p

                        image_file_str_sp = image_file_str.replace('\\', '/').split("/")[1]
                        image_file_str_sp = image_file_str_sp.split("@")
                        image_file_str_L = image_file_str_sp[0]
                        image_file_str_R = image_file_str_sp[1]
                        # print(image_file_str_L)
                        # print(image_file_str_R)
                        image_file_str_L_sp = image_file_str_L.split(",")
                        # print(image_file_str_L_sp)
                        if not int(image_file_str_L_sp[0]) == -99:

                            image_file_str_L_X = int(int(image_file_str_L_sp[0]) / (720 / IMG_WIDTH))
                            image_file_str_L_Y = int(int(image_file_str_L_sp[1]) / (720 / IMG_HEIGHT))
                            image_file_str_R_sp = image_file_str_R.split(",")
                            image_file_str_R_X = int(int(image_file_str_R_sp[0]) / (720 / IMG_WIDTH))
                            image_file_str_R_Y = int(int(image_file_str_R_sp[1]) / (720 / IMG_HEIGHT))
                            # print(image_file_str_L_sp)
                            # print(image_file_str_R_sp)

                            points_list = np.array(
                                [image_file_str_L_X, image_file_str_L_Y, image_file_str_R_X, image_file_str_R_Y])
                            # print(points_list)
                            # print(tar.shape)
                            # print(predict.shape)
                            # skeleton_img = example_input[:, :, :, 0:4]

                            example_points_list = points_list
                            X_bund = 0.13
                            Y_bund = 0.13
                            H = IMG_HEIGHT
                            W = IMG_WIDTH
                            X_cut_L = example_points_list[0]
                            Y_cut_L = example_points_list[1]

                            X_cut_R = example_points_list[2]
                            Y_cut_R = example_points_list[3]
                            tar_numpy = tar[index]
                            tar_numpy = torch.permute(tar_numpy, (1, 2, 0))
                            predict_numpy = predict[index]
                            predict_numpy = torch.permute(predict_numpy, (1, 2, 0))
                            hand_img = torch.ones([IMG_HEIGHT, IMG_WIDTH, 3])

                            shape_canvas_cut = hand_img[Y_cut_L if Y_cut_L > 0 else 0:int(Y_cut_L + Y_bund * 2 * H),
                                               X_cut_L if X_cut_L > 0 else 0:int(X_cut_L + X_bund * 2 * W), :].shape
                            # print(shape_canvas_cut)
                            # hand_L_img_cut = np.full([int(Y_bund * 2 * H), int(X_bund * 2 * W), 3], -1.)
                            # print(hand_L_img_cut.shape)
                            hand_L_img_cut = tar_numpy[
                                             Y_cut_L if Y_cut_L > 0 else 0:(
                                                                               Y_cut_L if Y_cut_L > 0 else 0) +
                                                                           shape_canvas_cut[
                                                                               0],
                                             X_cut_L if X_cut_L > 0 else 0:(
                                                                               X_cut_L if X_cut_L > 0 else 0) +
                                                                           shape_canvas_cut[
                                                                               1],
                                             :]
                            # hand_L_img_cut = torch.from_numpy(hand_L_img_cut).type(num_dtype)
                            hand_L_img_cut = torch.permute(hand_L_img_cut, [2, 0, 1])
                            hand_L_img_cut = resize_with_crop_or_pad_torch(hand_L_img_cut, 128, 128)
                            y_left_l = Y_cut_L if Y_cut_L > 0 else 0
                            y_right_l = (Y_cut_L if Y_cut_L > 0 else 0) + shape_canvas_cut[0]
                            x_left_l = X_cut_L if X_cut_L > 0 else 0
                            x_right_l = (X_cut_L if X_cut_L > 0 else 0) + shape_canvas_cut[1]
                            hand_L_img_cut_p = predict_numpy[y_left_l:y_right_l, x_left_l:x_right_l, :]
                            # hand_L_img_cut_p = torch.from_numpy(hand_L_img_cut_p).type(num_dtype)
                            hand_L_img_cut_p = torch.permute(hand_L_img_cut_p, [2, 0, 1])
                            hand_L_img_cut_p = resize_with_crop_or_pad_torch(hand_L_img_cut_p, 128, 128)

                            # hand_L_img[Y_cut_L if Y_cut_L > 0 else 0:int(Y_cut_L + Y_bund * 2 * H),
                            # X_cut_L if X_cut_L > 0 else 0:int(X_cut_L + X_bund * 2 * W),
                            # :] = tar_numpy[
                            #      Y_cut_L if Y_cut_L > 0 else 0:(Y_cut_L if Y_cut_L > 0 else 0) + shape_canvas_cut[0],
                            #      X_cut_L if X_cut_L > 0 else 0:(X_cut_L if X_cut_L > 0 else 0) + shape_canvas_cut[1],
                            #      :]
                            hand_img_b[:, 0:hand_L_img_cut.shape[1], 0:hand_L_img_cut.shape[2]] = hand_L_img_cut
                            hand_img_b[:, 0:hand_L_img_cut_p.shape[1], W - hand_L_img_cut_p.shape[2]:W
                            ] = hand_L_img_cut_p

                            # hand_R_img = np.ones([IMG_HEIGHT, IMG_WIDTH, 3])
                            shape_canvas_cut = hand_img[Y_cut_R if Y_cut_R > 0 else 0:int(Y_cut_R + Y_bund * 2 * H),
                                               X_cut_R if X_cut_R > 0 else 0:int(X_cut_R + X_bund * 2 * W), :].shape
                            # print(shape_canvas_cut)
                            # hand_R_img_cut = np.full([int(Y_bund * 2 * H), int(X_bund * 2 * W), 3], -1.)
                            # print(hand_R_img_cut.shape)
                            hand_R_img_cut = tar_numpy[
                                             Y_cut_R if Y_cut_R > 0 else 0:(
                                                                               Y_cut_R if Y_cut_R > 0 else 0) +
                                                                           shape_canvas_cut[
                                                                               0],
                                             X_cut_R if X_cut_R > 0 else 0:(
                                                                               X_cut_R if X_cut_R > 0 else 0) +
                                                                           shape_canvas_cut[
                                                                               1],
                                             :]

                            # hand_R_img_cut = torch.from_numpy(hand_R_img_cut).type(num_dtype)

                            hand_R_img_cut = torch.permute(hand_R_img_cut, [2, 0, 1])
                            hand_R_img_cut = resize_with_crop_or_pad_torch(hand_R_img_cut, 128, 128)
                            y_left_r = Y_cut_R if Y_cut_R > 0 else 0
                            y_right_r = (Y_cut_R if Y_cut_R > 0 else 0) + shape_canvas_cut[0]
                            x_left_r = X_cut_R if X_cut_R > 0 else 0
                            x_right_r = (X_cut_R if X_cut_R > 0 else 0) + shape_canvas_cut[1]
                            hand_R_img_cut_p = predict_numpy[y_left_r:y_right_r, x_left_r:x_right_r, :]
                            hand_R_img_cut_p = torch.permute(hand_R_img_cut_p, [2, 0, 1])
                            # hand_R_img_cut_p = torch.from_numpy(hand_R_img_cut_p).type(num_dtype)
                            hand_R_img_cut_p = resize_with_crop_or_pad_torch(hand_R_img_cut_p, 128, 128)
                            # hand_R_img[Y_cut_R if Y_cut_R > 0 else 0:int(Y_cut_R + Y_bund * 2 * H),
                            # X_cut_R if X_cut_R > 0 else 0:int(X_cut_R + X_bund * 2 * W),
                            # :] = tar_numpy[
                            #      Y_cut_R if Y_cut_R > 0 else 0:(Y_cut_R if Y_cut_R > 0 else 0) + shape_canvas_cut[0],
                            #      X_cut_R if X_cut_R > 0 else 0:(X_cut_R if X_cut_R > 0 else 0) + shape_canvas_cut[1],
                            #      :]
                            hand_img_b[:, H - hand_R_img_cut.shape[1]:H, 0:hand_R_img_cut.shape[2],
                            ] = hand_R_img_cut
                            hand_img_b[:, H - hand_R_img_cut_p.shape[1]:H, W - hand_R_img_cut_p.shape[2]:W
                            ] = hand_R_img_cut_p

                            # hand_img = torch.from_numpy(hand_img).type(num_dtype)
                            hand_img_list.append(hand_img_b)
                        else:

                            hand_img_list.append(hand_img_b)
                        index = index + 1

                    # hand_img_f = torch.stack(hand_img_list)
                    hand_img_list = np.array(hand_img_list)
                    # print("hand_img_list.shape:", hand_img_list.shape)
                    hand_img_f = torch.from_numpy(hand_img_list)
                    # print(hand_img_f.shape)
                    # hand_R_img_f = tf.stack(hand_R_img_list)
                    # torch.permute()
                    # input_image_b = example_input_image[:, 0:1, :, :]
                    input_body_and_fance_img = torch.concat(
                        [input_body_and_fance_img, input_body_and_fance_img, input_body_and_fance_img], dim=1)
                    # input_image_lh = example_input_image[:, 1:2, :, :]
                    input_left_hand_img = torch.concat([input_left_hand_img, input_left_hand_img, input_left_hand_img],
                                                       dim=1)
                    # input_image_rh = example_input_image[:, 2:3, :, :]
                    input_right_hand_img = torch.concat(
                        [input_right_hand_img, input_right_hand_img, input_right_hand_img], dim=1)
                    # input_image_f = example_input_image[:, 3:4, :, :]
                    # input_image_f = torch.concat([input_image_f, input_image_f, input_image_f], dim=1)
                    style_img = example_style_image
                    # print(torch.max(predict))
                    # print(torch.max(style_img))
                    # print(torch.max(tar))
                    # print(torch.min(predict))
                    # print(torch.min(style_img))
                    # print(torch.min(tar))
                    image = torch.concat(
                        [tar, input_body_and_fance_img, input_left_hand_img, input_right_hand_img, style_img, predict,
                         hand_img_f,
                         ], dim=3)
                    # print(image.shape)
                    image = (image + 1.0) / 2
                    image = torch.concat([image[i] for i in range(image.shape[0])], dim=1)

                    img_path = os.path.join('images_split_raw_cd', 'GAN-%d.jpg' % step_now)
                    # 保存
                    save_image(image, img_path)
                    val_step = val_step + 1
                    if val_step % validation_set_len == 0:
                        val_loader_iter = iter(validation_loader)
                    print(f"Step: {round(step_now / 1000, 1)}k")
                generator.train(True)
            # input_image, target, style_image , img_path
            # print(next(iter(training_loader)))
            input_image, target, style_image, img_path = next(train_loader_iter)  # inf_train_loader.next()
            # train_file_index=step_now%training_set_len
            # input_image, target, style_image, img_path = training_loader.__getitem__(train_file_index)

            # print(img_path)
            image_file_str_group = img_path
            hand_L_img_list = []
            hand_R_img_list = []
            hand_L_input_list = []
            hand_R_input_list = []
            index = 0
            bund_list = []

            head_bund_list = []
            head_img_list = []
            input_head_img_list = []

            need_train_D_hand = False
            file_index_name = ""
            npy_list = []

            processed_input_img_list = []
            for image_file_str in image_file_str_group:

                encoding = 'utf-8'
                # image_file_str = str(image_file_str, encoding)
                image_file_str_nodot = image_file_str.replace('\\', '/').split(".")[0]
                npflie = np.load("./" + image_file_str_nodot + ".npy")
                # print(npflie)

                nose_point = (npflie[53] * IMG_HEIGHT).astype(np.int32)

                # print(nose_point)
                image_file_str_L_X_N = nose_point[0] - 96
                image_file_str_L_Y_N = nose_point[1] - 96
                image_file_str_R_X_N = nose_point[0] + 96
                image_file_str_R_Y_N = nose_point[1] + 96

                X_cut_L_N = image_file_str_L_X_N
                Y_cut_L_N = image_file_str_L_Y_N

                X_cut_R_N = image_file_str_R_X_N
                Y_cut_R_N = image_file_str_R_Y_N

                tar_numpy = target[index]
                tar_numpy = torch.permute(tar_numpy, (1, 2, 0))

                input_body_img = input_image[index][0:1, :, :]
                input_left_hand_img = input_image[index][1:2, :, :]
                input_right_hand_img = input_image[index][2:3, :, :]
                input_face_img = input_image[index][3:4, :, :]
                input_body_and_fance_img = ((input_body_img + 1) + (input_face_img + 1)) - 1
                input_body_and_fance_img = torch.clip(input_body_and_fance_img, -1., 1.)
                # input_body_and_fance_img=input_body_img+input_face_img
                processed_input_img = torch.concat(
                    [input_body_and_fance_img, input_left_hand_img, input_right_hand_img], dim=0)
                processed_input_img_list.append(processed_input_img)

                input_body_and_fance_img = torch.permute(input_body_and_fance_img, (1, 2, 0))
                input_left_hand_img = torch.permute(input_left_hand_img, (1, 2, 0))
                input_right_hand_img = torch.permute(input_right_hand_img, (1, 2, 0))

                # head_img = np.ones([IMG_HEIGHT, IMG_WIDTH, 3])

                head_img_cut = tar_numpy[Y_cut_L_N:Y_cut_R_N, X_cut_L_N:X_cut_R_N, :]

                input_body_and_fance_img_cut = input_body_and_fance_img[Y_cut_L_N:Y_cut_R_N, X_cut_L_N:X_cut_R_N, :]
                input_left_hand_img_cut = input_left_hand_img[Y_cut_L_N:Y_cut_R_N, X_cut_L_N:X_cut_R_N, :]
                input_right_hand_img_cut = input_right_hand_img[Y_cut_L_N:Y_cut_R_N, X_cut_L_N:X_cut_R_N, :]

                head_img_cut = torch.permute(head_img_cut, [2, 0, 1])
                input_head_img_cut = torch.concat(
                    [input_body_and_fance_img_cut, input_left_hand_img_cut, input_right_hand_img_cut], dim=2)
                input_head_img_cut = torch.permute(input_head_img_cut, [2, 0, 1])
                # head_img_cut = resize_with_crop_or_pad_torch(head_img_cut, 128, 128)
                head_img_cut = torch_resize(head_img_cut)
                input_head_img_cut = torch_resize(input_head_img_cut)

                head_bund_list_item = [Y_cut_L_N, Y_cut_R_N, X_cut_L_N, X_cut_R_N]
                head_img_list.append(head_img_cut)
                input_head_img_list.append(input_head_img_cut)
                head_bund_list.append(head_bund_list_item)

                npy_list.append(npflie)
                # print("image_file_str:" + image_file_str)
                # print(image_file_str.replace('\\', '/').split("..")[1])
                image_file_str_sp = image_file_str.replace('\\', '/').split("/")[1]
                image_file_str_sp = image_file_str_sp.split("@")

                file_index_name = image_file_str_sp[3]
                image_file_str_L = image_file_str_sp[0]
                image_file_str_R = image_file_str_sp[1]
                image_file_mode = image_file_str_sp[2]
                # print(image_file_str_L)
                # print(image_file_str_R)
                image_file_str_L_sp = image_file_str_L.split(",")
                image_file_str_R_sp = image_file_str_R.split(",")
                if int(image_file_str_L_sp[0]) > -99 or int(image_file_str_L_sp[0]) > -99 or int(
                        image_file_str_R_sp[0]) > -99 or int(image_file_str_R_sp[1]) > -99:
                    need_train_D_hand = True
                    image_file_str_L_X = int(int(image_file_str_L_sp[0]) / (720 / IMG_WIDTH))
                    image_file_str_L_Y = int(int(image_file_str_L_sp[1]) / (720 / IMG_HEIGHT))
                    image_file_str_R_X = int(int(image_file_str_R_sp[0]) / (720 / IMG_WIDTH))
                    image_file_str_R_Y = int(int(image_file_str_R_sp[1]) / (720 / IMG_HEIGHT))

                    points_list = np.array(
                        [image_file_str_L_X, image_file_str_L_Y, image_file_str_R_X, image_file_str_R_Y])
                    # print(points_list)
                    # print(tar.shape)
                    # print(predict.shape)
                    # skeleton_img = example_input[:, :, :, 0:4]

                    example_points_list = points_list
                    X_bund = 0.13
                    Y_bund = 0.13
                    H = IMG_HEIGHT
                    W = IMG_WIDTH
                    X_cut_L = example_points_list[0]
                    Y_cut_L = example_points_list[1]

                    X_cut_R = example_points_list[2]
                    Y_cut_R = example_points_list[3]
                    tar_numpy = target[index]
                    tar_numpy = torch.permute(tar_numpy, (1, 2, 0))
                    # print(tar_numpy.shape)
                    # input_lh = input_image[index, :, :, 1:2]
                    # input_rh = input_image[index, :, :, 2:3]

                    hand_L_img = np.ones([IMG_HEIGHT, IMG_WIDTH, 3])

                    shape_canvas_cut = hand_L_img[Y_cut_L if Y_cut_L > 0 else 0:int(Y_cut_L + Y_bund * 2 * H),
                                       X_cut_L if X_cut_L > 0 else 0:int(X_cut_L + X_bund * 2 * W), :].shape
                    # print(shape_canvas_cut)
                    # hand_L_img_cut = np.full([int(Y_bund * 2 * H), int(X_bund * 2 * W), 3], -1.)
                    # print(hand_L_img_cut.shape)
                    hand_L_img_cut = tar_numpy[
                                     Y_cut_L if Y_cut_L > 0 else 0:(
                                                                       Y_cut_L if Y_cut_L > 0 else 0) +
                                                                   shape_canvas_cut[
                                                                       0],
                                     X_cut_L if X_cut_L > 0 else 0:(
                                                                       X_cut_L if X_cut_L > 0 else 0) +
                                                                   shape_canvas_cut[
                                                                       1],
                                     :]
                    input_hand_L_img_cut = input_left_hand_img[
                                           Y_cut_L if Y_cut_L > 0 else 0:(
                                                                             Y_cut_L if Y_cut_L > 0 else 0) +
                                                                         shape_canvas_cut[
                                                                             0],
                                           X_cut_L if X_cut_L > 0 else 0:(
                                                                             X_cut_L if X_cut_L > 0 else 0) +
                                                                         shape_canvas_cut[
                                                                             1],
                                           :]
                    # print("hand_L_img_cut:", hand_L_img_cut.shape)
                    # if image_file_mode == "LR":
                    #     hand_L_input_cut = input_lh[
                    #                        Y_cut_L if Y_cut_L > 0 else 0:(
                    #                                                          Y_cut_L if Y_cut_L > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          0],
                    #                        X_cut_L if X_cut_L > 0 else 0:(
                    #                                                          X_cut_L if X_cut_L > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          1],
                    #                        :]
                    # elif image_file_mode == "LL":
                    #     hand_L_input_cut = input_lh[
                    #                        Y_cut_L if Y_cut_L > 0 else 0:(
                    #                                                          Y_cut_L if Y_cut_L > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          0],
                    #                        X_cut_L if X_cut_L > 0 else 0:(
                    #                                                          X_cut_L if X_cut_L > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          1],
                    #                        :]
                    # elif image_file_mode == "RR":
                    #     hand_L_input_cut = input_rh[
                    #                        Y_cut_L if Y_cut_L > 0 else 0:(
                    #                                                          Y_cut_L if Y_cut_L > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          0],
                    #                        X_cut_L if X_cut_L > 0 else 0:(
                    #                                                          X_cut_L if X_cut_L > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          1],
                    #                        :]

                    # hand_L_img_cut = tf.cast(hand_L_img_cut, dtype=tf.float32)

                    # hand_L_img_cut = torch.from_numpy(hand_L_img_cut).type(num_dtype)
                    hand_L_img_cut = torch.permute(hand_L_img_cut, [2, 0, 1])
                    hand_L_img_cut = resize_with_crop_or_pad_torch(hand_L_img_cut, 128, 128)
                    input_hand_L_img_cut = torch.permute(input_hand_L_img_cut, [2, 0, 1])
                    input_hand_L_img_cut = torch.concat(
                        [input_hand_L_img_cut, input_hand_L_img_cut, input_hand_L_img_cut], dim=0)
                    input_hand_L_img_cut = resize_with_crop_or_pad_torch(input_hand_L_img_cut, 128, 128)
                    # hand_L_img_cut = tf.image.resize_with_crop_or_pad(hand_L_img_cut, 128, 128)
                    # hand_L_input_cut = tf.image.resize_with_crop_or_pad(hand_L_input_cut, 128, 128)

                    y_left_l = Y_cut_L if Y_cut_L > 0 else 0
                    y_right_l = (Y_cut_L if Y_cut_L > 0 else 0) + shape_canvas_cut[0]
                    x_left_l = X_cut_L if X_cut_L > 0 else 0
                    x_right_l = (X_cut_L if X_cut_L > 0 else 0) + shape_canvas_cut[1]
                    # hand_L_img[Y_cut_L if Y_cut_L > 0 else 0:int(Y_cut_L + Y_bund * 2 * H),
                    # X_cut_L if X_cut_L > 0 else 0:int(X_cut_L + X_bund * 2 * W),
                    # :] = tar_numpy[
                    #      Y_cut_L if Y_cut_L > 0 else 0:(Y_cut_L if Y_cut_L > 0 else 0) + shape_canvas_cut[0],
                    #      X_cut_L if X_cut_L > 0 else 0:(X_cut_L if X_cut_L > 0 else 0) + shape_canvas_cut[1],
                    #      :]
                    # hand_L_img[0:hand_L_img_cut.shape[0], 0:hand_L_img_cut.shape[1], :] = hand_L_img_cut

                    hand_R_img = np.ones([IMG_HEIGHT, IMG_WIDTH, 3])
                    shape_canvas_cut = hand_R_img[Y_cut_R if Y_cut_R > 0 else 0:int(Y_cut_R + Y_bund * 2 * H),
                                       X_cut_R if X_cut_R > 0 else 0:int(X_cut_R + X_bund * 2 * W), :].shape
                    # print(shape_canvas_cut)
                    # hand_R_img_cut = np.full([int(Y_bund * 2 * H), int(X_bund * 2 * W), 3], -1.)
                    # print(hand_R_img_cut.shape)
                    hand_R_img_cut = tar_numpy[
                                     Y_cut_R if Y_cut_R > 0 else 0:(
                                                                       Y_cut_R if Y_cut_R > 0 else 0) +
                                                                   shape_canvas_cut[
                                                                       0],
                                     X_cut_R if X_cut_R > 0 else 0:(
                                                                       X_cut_R if X_cut_R > 0 else 0) +
                                                                   shape_canvas_cut[
                                                                       1],
                                     :]

                    input_hand_R_img_cut = input_right_hand_img[
                                           Y_cut_R if Y_cut_R > 0 else 0:(
                                                                             Y_cut_R if Y_cut_R > 0 else 0) +
                                                                         shape_canvas_cut[
                                                                             0],
                                           X_cut_R if X_cut_R > 0 else 0:(
                                                                             X_cut_R if X_cut_R > 0 else 0) +
                                                                         shape_canvas_cut[
                                                                             1],
                                           :]

                    # if image_file_mode == "LR":
                    #     hand_R_input_cut = input_rh[
                    #                        Y_cut_R if Y_cut_R > 0 else 0:(
                    #                                                          Y_cut_R if Y_cut_R > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          0],
                    #                        X_cut_R if X_cut_R > 0 else 0:(
                    #                                                          X_cut_R if X_cut_R > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          1],
                    #                        :]
                    # elif image_file_mode == "LL":
                    #     hand_R_input_cut = input_lh[
                    #                        Y_cut_R if Y_cut_R > 0 else 0:(
                    #                                                          Y_cut_R if Y_cut_R > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          0],
                    #                        X_cut_R if X_cut_R > 0 else 0:(
                    #                                                          X_cut_R if X_cut_R > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          1],
                    #                        :]
                    # elif image_file_mode == "RR":
                    #     hand_R_input_cut = input_rh[
                    #                        Y_cut_R if Y_cut_R > 0 else 0:(
                    #                                                          Y_cut_R if Y_cut_R > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          0],
                    #                        X_cut_R if X_cut_R > 0 else 0:(
                    #                                                          X_cut_R if X_cut_R > 0 else 0) +
                    #                                                      shape_canvas_cut[
                    #                                                          1],
                    #                        :]

                    # hand_R_img_cut = tf.cast(hand_R_img_cut, dtype=tf.float32)
                    # hand_R_img_cut = torch.from_numpy(hand_R_img_cut).type(num_dtype)
                    hand_R_img_cut = torch.permute(hand_R_img_cut, [2, 0, 1])
                    hand_R_img_cut = resize_with_crop_or_pad_torch(hand_R_img_cut, 128, 128)
                    input_hand_R_img_cut = torch.permute(input_hand_R_img_cut, [2, 0, 1])
                    input_hand_R_img_cut = torch.concat(
                        [input_hand_R_img_cut, input_hand_R_img_cut, input_hand_R_img_cut], dim=0)
                    input_hand_R_img_cut = resize_with_crop_or_pad_torch(input_hand_R_img_cut, 128, 128)
                    # hand_R_img_cut = tf.image.resize_with_crop_or_pad(hand_R_img_cut, 128, 128)
                    # hand_R_input_cut = tf.image.resize_with_crop_or_pad(hand_R_input_cut, 128, 128)

                    y_left_r = Y_cut_R if Y_cut_R > 0 else 0
                    y_right_r = (Y_cut_R if Y_cut_R > 0 else 0) + shape_canvas_cut[0]
                    x_left_r = X_cut_R if X_cut_R > 0 else 0
                    x_right_r = (X_cut_R if X_cut_R > 0 else 0) + shape_canvas_cut[1]

                    # hand_R_img[Y_cut_R if Y_cut_R > 0 else 0:int(Y_cut_R + Y_bund * 2 * H),
                    # X_cut_R if X_cut_R > 0 else 0:int(X_cut_R + X_bund * 2 * W),
                    # :] = tar_numpy[
                    #      Y_cut_R if Y_cut_R > 0 else 0:(Y_cut_R if Y_cut_R > 0 else 0) + shape_canvas_cut[0],
                    #      X_cut_R if X_cut_R > 0 else 0:(X_cut_R if X_cut_R > 0 else 0) + shape_canvas_cut[1],
                    #      :]
                    # hand_R_img[0:hand_R_img_cut.shape[0], 0:hand_R_img_cut.shape[1], :] = hand_R_img_cut

                    # hand_L_img = tf.cast(hand_L_img, dtype=tf.float32)
                    # print("hand_L_img_cut:",hand_L_img_cut.shape)
                    hand_L_img_list.append(hand_L_img_cut)
                    hand_L_input_list.append(input_hand_L_img_cut)
                    # hand_R_img = tf.cast(hand_R_img, dtype=tf.float32)
                    hand_R_img_list.append(hand_R_img_cut)
                    hand_R_input_list.append(input_hand_R_img_cut)

                    # hand_L_input_list.append(hand_L_input_cut)
                    # hand_R_img = tf.cast(hand_R_img, dtype=tf.float32)
                    # hand_R_input_list.append(hand_R_input_cut)

                    bund_list_item = [y_left_l, y_right_l, x_left_l, x_right_l, y_left_r, y_right_r, x_left_r,
                                      x_right_r]
                    # print(np.abs(bund_list_item[1]-bund_list_item[0]))
                    # print(np.abs(bund_list_item[3]-bund_list_item[2]))
                    # print(np.abs(bund_list_item[5]-bund_list_item[4]))
                    # print(np.abs(bund_list_item[7]-bund_list_item[6]))

                    bund_list.append(bund_list_item)
                else:
                    bund_list_item = [0, 0, 0, 0, 0, 0, 0, 0]
                    bund_list.append(bund_list_item)
                index = index + 1

            # bund_list = np.array(bund_list)
            # bund_list = torch.from_numpy(bund_list).type(torch.int)
            # need_train_D_hand = torch.from_numpy(need_train_D_hand).type(torch.bool)
            # print(bund_list)
            # hand_L_img_list = np.array(hand_L_img_list)
            # print(hand_L_img_list.shape)
            # hand_L_img_f = torch.from_numpy(hand_L_img_list)
            if need_train_D_hand:
                hand_L_img_f = torch.stack(hand_L_img_list, dim=0)
                # print(hand_L_img_f.shape)
                # hand_R_img_list = np.array(hand_R_img_list)
                hand_R_img_f = torch.stack(hand_R_img_list, dim=0)
                # hand_R_img_f = torch.from_numpy(hand_R_img_list)

                hand_img_f = torch.concat([hand_L_img_f, hand_R_img_f], dim=0)

                hand_input_L_img_f = torch.stack(hand_L_input_list, dim=0)
                # print(hand_L_img_f.shape)
                # hand_R_img_list = np.array(hand_R_img_list)
                hand_input_R_img_f = torch.stack(hand_R_input_list, dim=0)
                # hand_R_img_f = torch.from_numpy(hand_R_img_list)

                hand_input_img_f = torch.concat([hand_input_L_img_f, hand_input_R_img_f], dim=0)

            npy_list = np.array(npy_list)
            npy_list = torch.from_numpy(npy_list).type(num_dtype)

            # head_img_list = np.array(head_img_list)
            # head_img_list = torch.from_numpy(head_img_list)

            head_img_list = torch.stack(head_img_list, dim=0)
            input_head_img_list = torch.stack(input_head_img_list, dim=0)

            processed_input_img_list = torch.stack(processed_input_img_list, dim=0)

            # print(npy_list.shape)
            # hand_L_input_f = tf.stack(hand_L_input_list)
            # # print(len(hand_L_input_list))
            # hand_R_input_f = tf.stack(hand_R_input_list)
            # hand_input_f = tf.concat([hand_L_input_f, hand_R_input_f], axis=0)
            # print(hand_img_f.shape)
            # print("step2")

            # gen_output = generator(input_image, training=True)
            #
            # gen_img = gen_output[0]
            # # print(gen_img.shape)
            # bund_list_item = bund_list[0]
            # gen_output_cut_list, gen_img_cut_l, gen_img_cut_r = cut_image(gen_img, bund_list_item)
            # hand_input_f
            # print(target.shape)
            # input_image, target, style_image
            if need_train_D_hand:
                gen_total_loss, gen_gan_loss_all, gen_gan_loss_hand, gen_gan_loss_head, gen_l1_loss, disc_loss, disc_loss_hand, disc_loss_head, glr, dlr, dhlr, dhelr, gen_img_cut_l, gen_img_cut_r, gen_img_cut_head, gen_img, LAMBDA = train_step(
                    processed_input_img_list, style_image, hand_img_f, hand_input_img_f,
                    bund_list, head_img_list, input_head_img_list, head_bund_list, npy_list, target, LAMBDA,
                    step_now)
            else:
                gen_total_loss, gen_gan_loss_all, gen_gan_loss_hand, gen_gan_loss_head, gen_l1_loss, disc_loss, disc_loss_hand, disc_loss_head, glr, dlr, dhlr, dhelr, gen_img_cut_head, gen_img, LAMBDA = train_step_nohand(
                    processed_input_img_list, style_image, target, head_img_list, input_head_img_list, head_bund_list,
                    npy_list, LAMBDA,
                    step_now)

            # Training step
            if (step_now + 1) % 10 == 0:
                # print('.', end='', flush=True)
                print("step:", step_now, "step_real:", step_real, " g_t_loss:", gen_total_loss,
                      " g_loss_all:",
                      gen_gan_loss_all,
                      " g_loss_hand:",
                      gen_gan_loss_hand,
                      " g_loss_head:",
                      gen_gan_loss_head,
                      " g_l1_loss:", gen_l1_loss,
                      # " g_s_loss:", loss_style,
                      " d_loss:", disc_loss,
                      " d_loss_hand:", disc_loss_hand,
                      " d_loss_head:", disc_loss_head,

                      " glr:", glr,
                      " dlr:", dlr,
                      " dhlr:", dhlr,
                      " dhelr:", dhelr,
                      "L:", LAMBDA,
                      )
                print(colored(f'Time taken for 10 steps: {time.time() - start_10:.2f} sec\n', 'green'))
                start_10 = time.time()

            if (step_now + 1) % 250 == 0 and need_train_D_hand:
                # input_input_f_l = tf.concat([hand_input_f[0], hand_input_f[0], hand_input_f[0]], axis=2)
                # input_input_f_r = tf.concat([hand_input_f[1], hand_input_f[1], hand_input_f[1]], axis=2)

                comp_img_l = torch.concat([gen_img_cut_l, hand_img_f[0], hand_input_img_f[0]], dim=1)
                comp_img_r = torch.concat([gen_img_cut_r, hand_img_f[1], hand_input_img_f[1]], dim=1)
                comp_img_head = torch.concat([gen_img_cut_head, head_img_list[0], input_head_img_list[0]], dim=1)
                comp_img_l = resize_with_crop_or_pad_torch(comp_img_l, IMG_HEIGHT, IMG_WIDTH)
                comp_img_r = resize_with_crop_or_pad_torch(comp_img_r, IMG_HEIGHT, IMG_WIDTH)
                comp_img_head = resize_with_crop_or_pad_torch(comp_img_head, IMG_HEIGHT, IMG_WIDTH)

                target_show = torch.reshape(target, [target.shape[1], target.shape[2], target.shape[3]])
                input_image_b = input_image[0, 0:1, :, :]
                # input_image_b = tf.concat([input_image_b, input_image_b, input_image_b], axis=3)
                input_image_lh = input_image[0, 1:2, :, :]
                # input_image_lh = tf.concat([input_image_lh, input_image_lh, input_image_lh], axis=3)
                input_image_rh = input_image[0, 2:3, :, :]
                # input_image_rh = tf.concat([input_image_rh, input_image_rh, input_image_rh], axis=3)
                input_image_f = input_image[0, 3:4, :, :]
                input_image_f = torch.concat([input_image_f, input_image_f, input_image_f], dim=0)
                input_image_f = (input_image_f + 1.0)
                input_image = torch.concat([input_image_b, input_image_lh, input_image_rh], dim=0)
                input_image = (input_image + 1.0)
                input_image = input_image + input_image_f
                input_image = (input_image - 1.0)
                image = torch.concat([comp_img_l, comp_img_r, comp_img_head, gen_img, target_show, input_image], dim=2)

                image = (image + 1.0) / 2
                # image = tf.concat([image[i] for i in range(image.shape[0])], axis=0)
                # image = tf.reshape([image], [image.shape[1], image.shape[2] * image.shape[0], image.shape[3]])
                # image = tf.squeeze(image, axis=0)
                # print(image.shape)

                img_path = os.path.join('images_split_raw_cd', f'GAN_train-%d_{file_index_name}.jpg' % step_now)
                # 保存
                save_image(image, img_path)
            elif (step_now + 1) % 250 == 0 and not need_train_D_hand:

                comp_img_head = torch.concat([gen_img_cut_head, head_img_list[0]], dim=1)

                comp_img_head = resize_with_crop_or_pad_torch(comp_img_head, IMG_HEIGHT, IMG_WIDTH)

                target_show = torch.reshape(target, [target.shape[1], target.shape[2], target.shape[3]])
                input_image_b = input_image[0, 0:1, :, :]
                # input_image_b = tf.concat([input_image_b, input_image_b, input_image_b], axis=3)
                input_image_lh = input_image[0, 1:2, :, :]
                # input_image_lh = tf.concat([input_image_lh, input_image_lh, input_image_lh], axis=3)
                input_image_rh = input_image[0, 2:3, :, :]
                # input_image_rh = tf.concat([input_image_rh, input_image_rh, input_image_rh], axis=3)
                input_image_f = input_image[0, 3:4, :, :]
                input_image_f = torch.concat([input_image_f, input_image_f, input_image_f], dim=0)
                input_image_f = (input_image_f + 1.0)
                input_image = torch.concat([input_image_b, input_image_lh, input_image_rh], dim=0)
                input_image = (input_image + 1.0)
                input_image = input_image + input_image_f
                input_image = (input_image - 1.0)
                image = torch.concat([comp_img_head, gen_img, target_show, input_image], dim=2)

                image = (image + 1.0) / 2
                # image = tf.concat([image[i] for i in range(image.shape[0])], axis=0)
                # image = tf.reshape([image], [image.shape[1], image.shape[2] * image.shape[0], image.shape[3]])
                # image = tf.squeeze(image, axis=0)
                # print(image.shape)

                img_path = os.path.join('images_split_raw_cd', f'GAN_train-%d_{file_index_name}.jpg' % step_now)
                # 保存
                save_image(image, img_path)

            # Save (checkpoint) the model every 5k steps
            if step_real % 40000 == 0 and step_real >= 100000:
                print("start saving model")
                model_path = checkpoint_dir + '/model_g_{}_{}.pt'.format(timestamp, step_now)
                torch.save(generator.state_dict(), model_path)
                model_path = checkpoint_dir + '/model_d_{}_{}.pt'.format(timestamp, step_now)
                torch.save(discriminator.state_dict(), model_path)
                model_path = checkpoint_dir + '/model_dh_{}_{}.pt'.format(timestamp, step_now)
                torch.save(discriminator_hand.state_dict(), model_path)

        except Exception as msg:
            print(msg)
            traceback.print_exc()

        step_now = step_now + 1


if __name__ == '__main__':
    freeze_support()
    print("start train")
    train_loop()
