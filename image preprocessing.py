import cv2
import os
import numpy as np

def apply_gamma_correction(image, gamma=1.2):
    """
    使用伽马校正来调整图像亮度，增强暗部细节。
    """
    # 将图像像素值归一化到0-1之间，进行伽马校正
    gamma_corrected = np.array(255 * (image / 255) ** (1 / gamma), dtype='uint8')
    return gamma_corrected

def denoise_image(image):
    """
    使用 Fast Non-Local Means 去噪算法对图像进行去噪。
    """
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

def sharpen_image(image, strength=1.5):
    """
    使用温和的锐化卷积核增强图像的细节，避免过度锐化导致失真。
    """
    # 温和的锐化卷积核，降低锐化的强度
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]]) * strength  # 锐化强度
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened

def process_images_in_folder(input_folder, output_folder, gamma=1.2, sharpen_strength=1.1):
    """
    处理文件夹中的所有图像，进行伽马校正、去噪、锐化并保存处理后的图像。
    """
    # 如果输出文件夹不存在，则创建一个
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 获取图像文件的完整路径
            image_path = os.path.join(input_folder, filename)

            # 读取图像
            image = cv2.imread(image_path)

            if image is not None:
                # 进行伽马校正（增强暗部）
                gamma_corrected_image = apply_gamma_correction(image, gamma)

                # 进行去噪处理
                denoised_image = denoise_image(gamma_corrected_image)

                # 进行锐化处理
                sharpened_image = sharpen_image(denoised_image, sharpen_strength)

                # 保存处理后的图像到输出文件夹
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, sharpened_image)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to load image: {image_path}")
        else:
            # 跳过非图像文件
            continue

if __name__ == '__main__':
    # 输入文件夹路径（包含待处理的图像）
    input_folder = r"C:\Users\86137\Desktop\Computer Vision\datasets\images\test"  # 修改为你的文件夹路径

    # 输出文件夹路径（保存处理后的图像）
    output_folder = r"C:\Users\86137\Desktop\Computer Vision\datasets\images\test_1"  # 修改为你的输出文件夹路径

    # 处理文件夹中的所有图像
    process_images_in_folder(input_folder, output_folder, gamma=1.2, sharpen_strength=1.1)  # 调整锐化强度
