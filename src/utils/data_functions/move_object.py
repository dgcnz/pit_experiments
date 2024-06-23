import torch
import random


def move_object_input_fn(num_samples, img_width, img_height):
    # Initialize the tensor to hold the batch of images
    batch_images = torch.zeros((num_samples, img_height, img_width))
    target_points = []

    for i in range(num_samples):
        # Randomly select a point for the top-left corner of the 3x3 object
        x = random.randint(0, img_width - 3)
        y = random.randint(0, img_height - 3)

        # Place the 3x3 object using 1s
        batch_images[i, y : y + 3, x : x + 3] = 1

        # Determine a random point for the object to move to
        # Ensuring it does not overlap with the original 3x3 object
        target_x, target_y = x, y
        while x <= target_x < x + 3 and y <= target_y < y + 3:
            target_x = random.randint(0, img_width - 1)
            target_y = random.randint(0, img_height - 1)

        # Set the target point (using a 1)
        batch_images[i, target_y, target_x] = 1
        target_points.append((target_x, target_y))

    return batch_images


def move_object_target_fn(input_images):
    output_images = torch.zeros_like(input_images)

    for i, img in enumerate(input_images):
        # Pad the image to handle edge cases, assuming padding with zero
        padded_img = torch.nn.functional.pad(
            img, (1, 1, 1, 1), mode="constant", value=0
        )

        # Calculate the number of 1's neighbors for each cell
        neighbors = (
            padded_img[1:-1, :-2]  # Left
            + padded_img[1:-1, 2:]  # Right
            + padded_img[:-2, 1:-1]  # Top
            + padded_img[2:, 1:-1]  # Bottom
        )

        # Identify the target point as having 1 or less neighbors
        target_point = (neighbors <= 1) & (img == 1)
        target_y, target_x = target_point.nonzero(as_tuple=True)

        # Handle the rare case of multiple potential target points
        if target_y.shape[0] > 1:
            # Heuristic: choose the first by default
            target_y, target_x = target_y[0], target_x[0]
        else:
            target_y, target_x = target_y.item(), target_x.item()

        # Find the original 3x3 object by checking for a block of `1`s
        object_mask = (img == 1) & ~target_point
        object_indices = object_mask.nonzero()
        if object_indices.numel() == 0:
            continue  # No object found, should handle appropriately
        min_y, min_x = object_indices[:, 0].min(), object_indices[:, 1].min()

        # Clear the space for the object in the output
        output_images[i, min_y : min_y + 3, min_x : min_x + 3] = (
            0  # Clear the original location in output
        )

        # Move the object to the target point
        output_images[i, target_y : target_y + 3, target_x : target_x + 3] = 1

    return output_images
