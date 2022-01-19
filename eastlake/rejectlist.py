import os
import yaml


class RejectList(object):
    """A class for storing a rejectlist - a list of images
    to be treated differently because, e.g., there may not exist a
    useful accompanying Piff file
    """

    def __init__(self, rejectlist_data):
        self.rejectlist_data = rejectlist_data

    @classmethod
    def from_file(cls, rejectlist_file):
        # read the rejectlist from the file
        with open(os.path.expandvars(rejectlist_file), "r") as fp:
            rejectlist_data = yaml.safe_load(fp)
        return cls(rejectlist_data)

    def img_file_is_rejectlisted(self, img_file):
        """
        Determine whether an image is in the rejectlist
        from its filename

        Parameters
        ----------
        img_file: str
            the image's filename

        Returns
        -------
        is_rejectlisted: bool
            whether or not the image is in the rejectlist
        """
        # Grab the exposure number and chip
        # number from the image filename
        img_file = os.path.basename(img_file)
        # image files have the format
        # "D<exp_num>_<band>_c<chip_num>_<other stuff>"
        exp_num = int(img_file.split("_")[0][1:])
        chip_num = int(img_file.split("_")[2][1:])
        is_rejectlisted = self.is_rejectlisted(exp_num, chip_num)
        return is_rejectlisted

    def is_rejectlisted(self, exp_num, chip_num):
        """Determine whether an image is in the rejectlist
        from its exp_num and chip_num
        """
        is_rejectlisted = (exp_num, chip_num) in self.rejectlist_data
        return is_rejectlisted
