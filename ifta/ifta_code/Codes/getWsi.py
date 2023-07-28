from tiffslide import TiffSlide
#import openslide


def getWsi(path):  # imports a WSI
    #wsi = openslide.OpenSlide(path)
    wsi = TiffSlide(path)
    return wsi
