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
    return images, poses, focal


def pose_encoder(x, hp_L=5):
    noise = [x]
    for i in range(hp_L):
        noise.append(np.cos(2 ** i * np.pi * x + 10 * i * np.pi))
        noise.append(np.sin(2 ** i * np.pi * x + 10 * i * np.pi))
    return torch.from_numpy(np.array(noise)).flatten()


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


# Stream Part 2


def get_ray(width, height, focal_length, camera_to_world):
    xs, ys = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
    )
    directions = torch.stack(
        (
            (xs - width * 0.5) / focal_length,
            -(ys - height * 0.5) / focal_length,
            -torch.ones_like(xs),
        ),
        -1,
    )
    rays_d = torch.sum(directions[..., np.newaxis, :] * camera_to_world[:3, :3], -1)
    rays_o = torch.broadcast_to(camera_to_world[:3, -1], rays_d.shape)
    return rays_o, rays_d


def render(near, far, num_samples, ray_origin, ray_direction, nerf):
    # Sample rays from near to far.
    pts = torch.linspace(near, far, num_samples)
    # Sample one random point, scale it and and add it to the current point. This randomizations
    # offsets the point to somewhere between near pt and pt + 1
    pts += torch.rand((ray_origin.shape[:-1], num_samples)) * (far - near) / num_samples

    # Stream part 3.
    # Sampling new hit points for density and radiance.
    sample_pts = (
        ray_origin[..., None, :] + ray_direction[..., None, :] * pts[..., :, None]
    )
    samples = sample_pts.view(-1, 3)
    pos = pose_encoder(samples, 5)  # Paper uses 10.
    dirs = pose_encoder(sample_pts, 4)
    density, radiance = nerf(dirs, pos)

    # Volume rendering
    dists = pts[..., 1:] - pts[..., :-1]
    alpha = 1 - torch.exp(-density * dists)
    weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, -1)
    rgb_color = torch.sum(weights * radiance, -2)

    return rgb_color


def train(images, poses, focal, height, width):
    num_samples = 4
    num_iters = 1000
    psnr = []
    nerf = Nerf()

    optimizer =  torch.optim.Adam(nerf.parameters(), lr=1e-5)

    for i in range(num_iters + 1):
        rnd_i = np.random.randint(images.shape[0])
        target = images[rnd_i]
        pose = poses[rnd_i]

        optimizer.zero_grad()
        ray_o, ray_d = get_ray(width, height, focal, pose)
        rgb = render(2.0, 6.0, num_samples, ray_o, ray_d, nerf)
        loss = torch.nn.MSELoss()(rgb, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Printing stats:\n Step: {}".format(i))
            # TODO: Use the test data to plot stats

def test():
    image, poses, focal = load_and_view_data()
    print(image.shape)
    rays_o, rays_d = get_ray(image[0].shape[1], image[0].shape[0], focal, poses[0])
    print("Directions:\n", rays_d)
    print(rays_d.shape)
