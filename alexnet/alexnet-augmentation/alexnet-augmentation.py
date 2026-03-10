import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    # YOUR CODE HERE
    h, w = image.shape[:2]
    if crop_size > h or crop_size > w:
        return image
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)

    return image[top:top + crop_size, left:left + crop_size]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally."""
    # YOUR CODE HERE
    if np.random.rand() < p:
        flipped_img = np.flip(image, axis=1)
        assert np.array_equal(flipped_img, image[:, ::-1, :]), "expect 2 approaches are same"
        return flipped_img
    return image