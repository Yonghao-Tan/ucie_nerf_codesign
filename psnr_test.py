import cv2
import numpy as np

def calculate_psnr(img1_path, img2_path):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
    
    Returns:
        float: PSNR value in decibels (dB).
    """
    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Ensure both images are loaded
    if img1 is None or img2 is None:
        raise ValueError("One or both image paths are invalid or the images cannot be loaded.")

    # Resize img1 to match img2 dimensions if sizes are different
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert to float for MSE calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR for identical images
    
    # Calculate PSNR
    max_pixel_value = 255.0  # Assuming 8-bit images
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr


# Example usage
if __name__ == "__main__":
    # Define file paths for the three comparisons
    base_path_test = "/home/ytanaz/access/IBRNet/eval/llff_test/eval_llff_8/"
    base_path = "/home/ytanaz/access/IBRNet/eval/llff_test/eval_llff_8/fern_000000/"
    image_pairs = [
        ("image000_pred_fine.png", "image000_gt_rgb.png"),
        ("image008_pred_fine.png", "image008_gt_rgb.png"),
        ("image016_pred_fine.png", "image016_gt_rgb.png")
    ]
    image_pairs = [
        ("image000_pred_fine_x2_DAT.png", "image000_gt_rgb.png"),
        ("image008_pred_fine_x2_DAT.png", "image008_gt_rgb.png"),
        ("image016_pred_fine_x2_DAT.png", "image016_gt_rgb.png")
    ]

    # Calculate PSNR for each pair
    psnr_results = []
    for pred, gt in image_pairs:
        img1_path = base_path_test + pred
        img2_path = base_path + gt
        psnr_value = calculate_psnr(img1_path, img2_path)
        psnr_results.append(psnr_value)
    
    # print(f"PSNR between the two images: {psnr_value:.2f} dB")
    print(f"{psnr_results}")
