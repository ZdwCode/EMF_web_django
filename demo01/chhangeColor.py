from PIL import Image
image = Image.open('./static/img/green.jpg')
im_new = Image.new(image.mode, image.size);
width, height = image.size;
for i in range(0, width):
    for j in range(0, height):
        if image.getpixel(xy=(i, j)) != (255, 255, 255):
            image.putpixel(xy=(i, j), value=(0, 255, 0))
        # im_new.putpixel(xy=(i, j), value=image.getpixel(xy=(i, j)));
image.show()