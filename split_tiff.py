from PIL import Image, ImageSequence

im = Image.open("training_data_cropped/2025107_R8_1to500to100_cell4_208_cropped.tif")

for i, page in enumerate(ImageSequence.Iterator(im)):
    page.save("208_%d.png" % i)