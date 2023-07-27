from tiffslide import TiffSlide


def getWsi(path):  # imports a WSI
    wsi = TiffSlide(path)
    return wsi
