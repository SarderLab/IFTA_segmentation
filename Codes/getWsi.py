from pathlib import Path
def getWsi(path: str):
    path = str(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WSI path does not exist in container: {path}")

    try:
        from tiffslide import TiffSlide
        slide = TiffSlide(path)
        return slide
    except Exception as tiff_err:
        try:
            import openslide
            slide = openslide.OpenSlide(path)
            return slide
        except openslide.OpenSlideError as os_err:
            raise RuntimeError(
                f"Neither TiffSlide nor OpenSlide can open '{path}'. "
                "Check that the file is a valid WSI and mounted correctly in the container."
            ) from os_err