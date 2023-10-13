import subprocess
import numpy as np
import fitsio
import yaml

# unload the seeds
try:
    subprocess.run(
        "rm -f meds_seeds.fits",
        shell=True,
        check=True,
    )
    subprocess.run(
        "easyaccess --db desoper -c 'select * from gruendl.prelim_meds_compression_seed; > meds_seeds.fits'",
        shell=True,
        check=True,
    )

    fd = fitsio.read("meds_seeds.fits", lower=True)
    d = {}
    for tname in np.unique(fd["tilename"]).tolist():
        d[tname] = {}
        d[tname]["pfw_attempt_id"] = {}
        for band in ["g", "r", "i", "z"]:
            msk = (fd["tilename"] == tname) & (fd["band"] == band)
            if not np.any(msk):
                continue
            d[tname][band] = {}
            for ext, col in [
                ("image_cutouts", "img_seed"),
                ("weight_cutous", "wgt_seed"),
                ("psf_cutouts", "psf_seed"),
            ]:
                d[tname][band][ext] = int(fd[col][msk][0])

            d[tname]["pfw_attempt_id"][band] = str(fd["pfw_attempt_id"][msk][0])

    destfile = "../eastlake/config/Y6A1_v1_meds-desdm-Y6A1v11-fpack-seeds.yaml"
    with open(destfile, "w") as fp:
        yaml.safe_dump(d, fp)
finally:
    subprocess.run(
        "rm -f meds_seeds.fits",
        shell=True,
        check=True,
    )
