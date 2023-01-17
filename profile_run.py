from cProfile import Profile
from pstats import SortKey, Stats

from mammography.src.data import extract_dicom
from mammography.src.dicomsdl import process_dicom
from mammography.src.preprocess_images import process_image

# fn = process_dicom
fn = lambda filepath: list(extract_dicom(filepath))

bad_bit_depth = "mammography/data/raw/train_images/32553/83977.dcm"
one_window_null = "mammography/data/raw/train_images/29927/68491.dcm"
multi_window_sigmoid_invert = "mammography/data/raw/train_images/10868/225973.dcm"

if __name__ == "__main__":
    with Profile() as p:
        fn(bad_bit_depth)
        fn(one_window_null)
        fn(multi_window_sigmoid_invert)
    stats = Stats(p)
    stats = stats.strip_dirs()
    stats = stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10)
