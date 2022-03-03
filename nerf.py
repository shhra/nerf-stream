import numpy as np
import matplotlib.pyplot as plt
import torch


np.random.seed(3032022)

# To download the data use follow command:
# wget http://cseweb.ucsd.edu/\~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz


# STREAM PART 1
def load_and_view_data(visualize=False):
    data = np.load("tiny_nerf_data.npz", allow_pickle=False)
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    if visualize:
        random_id = np.random.randint(0, 100)
        random_image, random_pose = images[random_id], poses[random_id]
        print("Camera pose data:\n {}".format(random_pose))
        print("Focal data: {}".format(focal))
        plt.imshow(random_image)
        plt.show()


def pose_encoder(x, hp_L=5):
    noise = [x]
    for i in range(hp_L):
        noise.append(np.cos(2 ** i * np.pi * x + 10 * i * np.pi))
        noise.append(np.sin(2 ** i * np.pi * x + 10 * i * np.pi))
    return np.array(noise)


class Nerf(torch.nn.Module):
    """This is totaly random math invented by a lazy person who wants to GET. SHIT. DONE.
    Takes input gives "volume density"
    """
    def __init__(self, num_layers=8, dim=256, hp_L=5):
        super(Nerf, self).__init__()
        layers = []
        input_size = 3 + 3 * 2 * hp_L
        # First block
        layers += [torch.nn.Linear(input_size, dim)]
        layers += [torch.nn.ReLU()]
        active = 0
        for i in range(1, num_layers):
            if i % 4 == 0:
                active = i
                break
            layers += [torch.nn.Linear(dim, dim)]
            layers += [torch.nn.ReLU()]
        self.first_block = torch.nn.Sequential(*layers)

        layers = [torch.nn.Linear(dim + input_size, dim)]
        for i in range(active, num_layers):
            layers += [torch.nn.Linear(dim, dim)]
            layers += [torch.nn.ReLU()]
        self.second_block = torch.nn.Sequential(*layers)
        self.direction_layer = torch.nn.Linear(dim + 24, dim)
        self.final_layer = torch.nn.Linear(dim, dim // 2)
        self.color_output = torch.nn.Linear(dim // 2, 4)

    def forward(self, direction, position):
        y = self.first_block(position)
        y = self.second_block(torch.concat((y, position)))
        y = self.direction_layer(torch.concat((direction, y)))
        density = torch.nn.ReLU()(y)
        y = torch.nn.ReLU()(self.final_layer(density))
        radiance = torch.nn.Sigmoid()(self.color_output(y))
        return density, radiance