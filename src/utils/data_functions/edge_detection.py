import torch
import random


def edge_detection_batch_fn(num_samples, img_width, img_height):
    batch_images = torch.zeros((num_samples, img_height, img_width), dtype=torch.int)
    labels = torch.zeros(num_samples, dtype=torch.int)

    for i in range(num_samples):
        feature_type = random.choice(["line", "edge"])

        if feature_type == "line":
            labels[i] = 0
            if random.choice(["horizontal", "vertical"]) == "horizontal":
                row = random.randint(0, img_height - 1)
                start_col = random.randint(0, img_width - 3)
                batch_images[i, row, start_col : start_col + 3] = 1
            else:
                col = random.randint(0, img_width - 1)
                start_row = random.randint(0, img_height - 3)
                batch_images[i, start_row : start_row + 3, col] = 1
        elif feature_type == "edge":
            labels[i] = 1
            # Choose a starting point with enough room for an L-shape
            start_row = random.randint(0, img_height - 2)
            start_col = random.randint(0, img_width - 2)
            # Randomly choose the orientation of the L-shape
            orientation = random.choice(
                ["top-left", "top-right", "bottom-left", "bottom-right"]
            )

            if orientation == "top-left":
                batch_images[i, start_row, start_col] = 1
                batch_images[i, start_row + 1, start_col] = 1
                batch_images[i, start_row, start_col + 1] = 1
            elif orientation == "top-right":
                batch_images[i, start_row, start_col + 1] = 1
                batch_images[i, start_row + 1, start_col + 1] = 1
                batch_images[i, start_row, start_col] = 1
            elif orientation == "bottom-left":
                batch_images[i, start_row + 1, start_col] = 1
                batch_images[i, start_row, start_col] = 1
                batch_images[i, start_row + 1, start_col + 1] = 1
            elif orientation == "bottom-right":
                batch_images[i, start_row + 1, start_col + 1] = 1
                batch_images[i, start_row, start_col + 1] = 1
                batch_images[i, start_row + 1, start_col] = 1

    return batch_images, labels
