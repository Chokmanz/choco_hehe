from matplotlib.image import imread
import matplotlib.pyplot as plot
import numpy as np

def skibidi_value(scale_choice):
    global skibidi
    skibidi=scale_choice

def compress_image(image):
    plot.rcParams['figure.figsize'] = [19.2, 10.08]

    def grayscale():
        x = imread(image)
        grayx = np.mean(x, -1)
        pic = plot.imshow(grayx)
        pic.set_cmap('gray')
        plot.title('Gray-scaled image')
        plot.axis('off')
        plot.show()
        
        U, Σ, VT = np.linalg.svd(grayx, full_matrices=False)
        Σ = np.diag(Σ)
        z = 0
        for n in (5, 20, 100, 200, 500, 750, 1000):
            grayxapprox = U[:, :n] @ Σ[0:n, :n] @ VT[:n, :]
            plot.figure(z + 1)
            z += 1
            pic = plot.imshow(grayxapprox)
            pic.set_cmap('gray')
            plot.axis('off')
            plot.title('rank = ' + str(n))
            plot.show()
        plot.figure(2)
        plot.plot(np.cumsum(np.diag(Σ))/np.sum(np.diag(Σ)))
        plot.title('Rank - Approximation Percentage relation')
        plot.show()
    def rgb():
        x = imread(image)
        plot.imshow(x)
        plot.title('Original Image')
        plot.axis('off')
        plot.show()

        red_U, red_Σ, red_VT = np.linalg.svd(x[:, :, 0], full_matrices=False)
        green_U, green_Σ, green_VT = np.linalg.svd(x[:, :, 1], full_matrices=False)
        blue_U, blue_Σ, blue_VT = np.linalg.svd(x[:, :, 2], full_matrices=False)

        red_Σ = np.diag(red_Σ)
        green_Σ = np.diag(green_Σ)
        blue_Σ = np.diag(blue_Σ)
        z = 0

        for n in (5, 20, 100, 200, 500, 750, 1000):
            redapprox = red_U[:, :n] @ red_Σ[0:n, :n] @ red_VT[:n, :]
            greenapprox = green_U[:, :n] @ green_Σ[0:n, :n] @ green_VT[:n, :]
            blueapprox = blue_U[:, :n] @ blue_Σ[0:n, :n] @ blue_VT[:n, :]
            final = np.stack([redapprox, greenapprox, blueapprox], axis=-1)
            final = np.clip(final, 0, 255)
            plot.figure(z + 1)
            z += 1
            plot.imshow(final.astype(np.uint8))
            plot.axis('off')
            plot.title('rank = ' + str(n))
            plot.savefig(f'plot_rank_{n}.jpg', format='jpg')
            plot.show()

    
    if skibidi == 1:
        grayscale()
    elif skibidi == 2:
        rgb()
    else:
        print("Invalid input, please try again")