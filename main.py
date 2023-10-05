import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import math
from scipy.stats import kurtosis
from skimage import data
import matplotlib.pyplot as plt

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def thinning(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


def structural_statistical_text(letras):
    image_folder = "MAIUSCULAS"

    results = []

    for letra in letras:
        for i in range(1, 101):
            archive_name = f"{letra}{i:05d}.pgm"
            image_path = os.path.join(image_folder, archive_name)

            if os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                resized_image = cv2.resize(image, (200, 200))

                _, binarized = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY_INV)

                skeleton = thinning(binarized)

                contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_area = cv2.contourArea(contours[0]) if contours else 0

                skeleton_mean = np.mean(skeleton)

                mean = np.mean(image)
                std_deviation = np.std(image)
                curtose = kurtosis(image.flatten())

                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                hist /= hist.sum()
                entropy = -np.sum(np.multiply(hist, np.log2(hist + np.finfo(float).eps)))

                results.append([skeleton_mean, contours_area, mean, std_deviation, curtose, entropy])

    # Escreva os resultados no arquivo "results_statistical_structural.txt"
    with open("results_statistical_structural.txt", 'w') as f:
        for values in results:
            f.write(' '.join(map(str, values)) + '\n')


def glcm_values(image):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    return dissimilarity, homogeneity, contrast, asm, energy


def process_images_for_glcm(letters):
    glcm_values_list = []

    # Chame a função que preenche results_statistical_structural.txt
    structural_statistical_text(letters)

    # Leia os resultados do arquivo
    with open("results_statistical_structural.txt", "r") as result_file:
        structural_results = [line.strip().split() for line in result_file]

    for letter in letters:
        for i in range(1, 101):
            archive_name = f"{letter}{i:05d}.pgm"
            image_path = os.path.join("MAIUSCULAS", archive_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(image, (200, 200))
                values = glcm_values(resized_image)

                values_list = [float(val) for val in values]

                # Combine valores de glcm_values, structural_results e letter
                combined_values = values_list + structural_results.pop(0) + [letter]
                glcm_values_list.append(combined_values)

            else:
                print(f"Arquivo {archive_name} não encontrado.")

    with open("glcm_metrics.txt", 'w') as f:
        for values in glcm_values_list:
            f.write(f"{values[0]} {values[1]} {values[2]} {values[3]} {values[4]} {values[5]} {' '.join(map(str, values[6:]))}\n")

def main():
    pass


def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    max_  = np.max(glcm, axis=(0,1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent


if __name__ == '__main__':
    main()

    levels = 8
    ks = 5
    mi, ma = 0, 255

    img = data.camera()
    h,w = img.shape

    img[:,:w//2] = img[:,:w//2]//2+127
    glcm_mean = fast_glcm_mean(img, mi, ma, levels, ks)

def glcm_processing(letra):
    image_folder = "MAIUSCULAS"
    result_folder = "RESULTADO_GLCM"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for i in range(1, 11):
        archive_name = f"{letra}{i:05d}.pgm"
        image_path = os.path.join(image_folder, archive_name)

        if os.path.exists(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            resized_image = cv2.resize(image, (200, 200))

            dissimilarity = fast_glcm_dissimilarity(resized_image)
            homogeneity = fast_glcm_homogeneity(resized_image)
            contrast = fast_glcm_contrast(resized_image)
            ASM_value, energy_value = fast_glcm_ASM(resized_image)

            dissimilarity_scaled = (
                (dissimilarity - dissimilarity.min())
                / (dissimilarity.max() - dissimilarity.min())
                * 255
            )

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f"Métricas GLCM para {letra}{i:05d}", fontsize=16)

            axes[0, 0].imshow(resized_image, cmap="gray")
            axes[0, 0].set_title("Imagem Original")

            axes[0, 1].imshow(np.uint8(dissimilarity_scaled), cmap="gray")
            axes[0, 1].set_title("Dissimilarity")

            axes[0, 2].imshow(np.uint8(homogeneity * 255), cmap="gray")
            axes[0, 2].set_title("Homogeneity")

            axes[1, 0].imshow(np.uint8(contrast * 255), cmap="gray")
            axes[1, 0].set_title("Contrast")

            axes[1, 1].imshow(np.uint8(ASM_value * 255), cmap="gray")
            axes[1, 1].set_title("ASM")

            axes[1, 2].imshow(np.uint8(energy_value * 255), cmap="gray")
            axes[1, 2].set_title("Energy")

            for ax in axes.flatten():
                ax.axis("off")

            plt.savefig(
                os.path.join(
                    result_folder, f"{letra}{i:05d}_metricas_glcm.png"), dpi=300
            )

            plt.close(fig)
        else:
            print(f"Arquivo {archive_name} não encontrado.")


def structural_and_statistical_processing(letra):
    image_folder = "MAIUSCULAS"
    result_folder = "RESULTADO_ESTRUTURAIS_ESTATISTICAS"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for i in range(1, 26):
        archive_name = f"{letra}{i:05d}.pgm"
        image_path = os.path.join(image_folder, archive_name)

        if os.path.exists(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            resized_image = cv2.resize(image, (200, 200))

            _, binarized = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY_INV)

            skeleton = thinning(binarized)

            contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_image = np.zeros_like(resized_image)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

            resultado = cv2.hconcat([skeleton, contours_image])

            mean = np.mean(image)
            std_deviation = np.std(image)
            curtose = kurtosis(image.flatten())

            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist /= hist.sum()
            entropy = -np.sum(np.multiply(hist, np.log2(hist + np.finfo(float).eps)))

            text = f"\nLetra: {letra}\nMedia: {mean:.2f}\nDesvio Padrao: {std_deviation:.2f}\nCurtose: {curtose:.2f}\nEntropia: {entropy:.2f}"

            result_with_text = np.zeros((335, 400), dtype=np.uint8)
            result_with_text[:200, :] = resultado

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_lines = text.split('\n')
            for j, line in enumerate(text_lines):
                cv2.putText(result_with_text, line, (10, 220 + j * 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(result_folder, f"{letra}{i:05d}_skeleton_contours.png"), result_with_text)
        else:
            print(f"Arquivo {archive_name} não encontrado.")


# Chame a função process_images_for_glcm
#process_images_for_glcm(letters)


for letter in letters:
    structural_and_statistical_processing(letter)
    glcm_processing(letter)
    pass

