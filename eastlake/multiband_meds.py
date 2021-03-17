from __future__ import print_function
import galsim
from galsim.config.output import OutputBuilder
from galsim.des.des_meds import MEDSBuilder, MultiExposureObject
from galsim.noise import GaussianNoise


class MultibandMEDSBuilder(MEDSBuilder):
    """
    Output class to simulate MEDS files in multiple bands.
    Mostly, this just means making sure the same objects are repeated bewteen bands.
    As well as the arguments for the output base class (file_name, dir etc.), it has required arguments
    @param bands                List of band names e.g. ['r','i']
    @param nstamps_per_object   Number of epochs per object (per band)
    and optional arguments
    @param ntiles               Number of "tiles" such that total number of output files will be ntiles * len(bands)
    nfiles is not used by this class, and in fact providing raises an error to avoid confusion.
    @param psf_noise            Dictionary with keys s2n and rng_num to add noise to psf image
    """

    def setup(self, config, base, file_num, logger):
        logger.debug('Start MultibandMEDSBuilder setup file_num=%d', file_num)

        # Make sure tile_num and band_num are considered valid index_keys.
        # Also add tile_start_obj_num to the eval_variables,
        if 'tile_num' not in galsim.config.process.valid_index_keys:
            galsim.config.valid_index_keys += ['tile_num', 'band_num']
            galsim.config.eval_base_variables += ['tile_num', 'band_num',
                                                  'tile_start_obj_num', 'band', "tilename", "nbands"]

        if 'ntiles' in config:
            # Sometimes this will be called prior to ProcessInput being called, so if there is an
            # error, try loading the inputs and then try again.
            try:
                ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
            except Exception:
                galsim.config.ProcessInput(base, safe_only=True)
                ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
        else:
            ntiles = 1

        # To avoid ambiguity, don't allow the use of nfiles for this output class
        try:
            assert ("nfiles" not in config)
        except AssertionError:
            raise ValueError("Do not use nfiles for output type 'MultibandMEDS'")

        # We'll be setting the random number seed to repeat for each band, which requires
        # querying the number of objects in the exposure.  This however leads to a logical
        # infinite loop if the number of objects is a random variate.  So to make this work,
        # enforce that nobjects is a fixed integer.
        if 'nobjects' not in config:
            raise ValueError("nobjects is required for output type 'MultibandMEDS'")
        nobj = config['nobjects']
        if not isinstance(nobj, int):
            raise ValueError(
                "nobjects is required to be a fixed integer for type 'MultibandMEDS'")

        # Set the random numbers to repeat for the objects so we get the same objects in the field
        # each time. In fact what we do is generate three sets of random seeds:
        # 0 : Sequence of seeds that iterates with obj_num i.e. no repetetion. Used for noise
        # 1 : Sequence of seeds that starts with the first object number for a given tile, then iterates
        # with the obj_num minus the first object number for that band, intended for quantities
        # that should be the same between bands for a given tile.
        # 2: Sequence of seeds that iterates with tile_num

        rs = base['image']['random_seed']
        if not isinstance(rs, list):
            first = galsim.config.ParseValue(base['image'], 'random_seed', base, int)[0]
            try:
                assert (first != 0)
            except AssertionError as e:
                print(
                    "image.random_seed must be set to a non-zero integer for output type 'MultibandMEDS'")
                raise(e)
            base['image']['random_seed'] = []
            # The first one is the original random_seed specification, used for noise, since
            # that should be different for each band, and probably most things in input, output,
            # or image.
            if isinstance(rs, int):
                base['image']['random_seed'].append(
                    {'type': 'Sequence', 'index_key': 'obj_num', 'first': first})
            else:
                base['image']['random_seed'].append(rs)

            # The second one is used for the galaxies and repeats through the same set of seed
            # values for each band in a tile.
            if nobj > 0:
                base['image']['random_seed'].append(
                    {
                        'type': 'Eval',
                        'str': 'first + tile_start_obj_num + (obj_num -start_obj_num) // nstamps_per_object',
                        'ifirst': first,
                        'instamps_per_object': {'type': 'Current', 'key': 'output.nstamps_per_object'},
                    }
                )
            else:
                base['image']['random_seed'].append(base['image']['random_seed'][0])

            # The third iterates per tile
            base['image']['random_seed'].append(
                {'type': 'Sequence', 'index_key': 'tile_num', 'first': first})

            if 'gal' in base:
                base['gal']['rng_num'] = 1
            if 'star' in base:
                base['star']['rng_num'] = 1
            if 'stamp' in base:
                base['stamp']['rng_num'] = 1
            if 'image_pos' in base['image']:
                base['image']['image_pos']['rng_num'] = 1
            if 'world_pos' in base['image']:
                base['image']['world_pos']['rng_num'] = 1

        logger.debug('random_seed = %s', galsim.config.CleanConfig(
            base['image']['random_seed']))

        bands = config["bands"]
        nbands = len(bands)
        base['nbands'] = nbands

        # Make sure that tile_num and band_num are setup properly in the right places.
        # This is fairly simple for this output type, since the number of files is just ntiles x nbands.
        tile_num = file_num // nbands
        band_num = file_num % nbands

        # Add some useful variables to the base config
        base['tile_num'] = tile_num
        base['band_num'] = band_num
        base['band'] = config['bands'][band_num]

        nobjects = galsim.config.ParseValue(config, 'nobjects', base, int)[0]
        nepochs_per_object = galsim.config.ParseValue(
            config, 'nstamps_per_object', base, int)[0]

        # tile_start_obj_num is the object number of the first object in the current tile
        # for multiband-meds, this is the following.
        base['tile_start_obj_num'] = base['start_obj_num'] - \
            band_num * nobjects * nepochs_per_object

        # I think this may get screwed up if nstamps_per_object varies between bands, so for now, makes sure
        # its a fixed integer
        assert isinstance(config["nstamps_per_object"], int)

        # Set the tilename
        if 'tilename' in config:
            tilename = galsim.config.ParseValue(config, 'tilename', base, str)[0]
        else:
            tilename = "tile-%d" % tile_num
        base["tilename"] = tilename

        logger.info('file_num, ntiles, nband = %d, %d, %d', file_num, ntiles, nbands)
        logger.info('tile_num, band_num = %d, %d', tile_num, band_num)
        logger.info("start_obj_num = %d", base['start_obj_num'])
        logger.info("tile_start_obj_num = %d", base['tile_start_obj_num'])

        # This sets up the RNG seeds.
        OutputBuilder.setup(self, config, base, file_num, logger)

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """
        Build a meds file as specified in config.
        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file_num.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in config that we can
                                ignore here.
        @param logger           If given, a logger object to log progress.
        @returns obj_list
        """

        logger.info('Starting buildImages')
        logger.info('file_num: %d' % base['file_num'])
        logger.info('image_num: %d', base['image_num'])

        tile_num = base['tile_num']
        band_num = base['band_num']
        req = {'nbands': int, 'nobjects': int, 'nstamps_per_object': int}
        opt = {'ntiles': int, 'tilename': str, 'psf_s2n': float}
        ignore += ['file_name', 'dir', 'bands', 'tilename', 'nfiles']
        params = galsim.config.GetAllParams(
            config, base, req=req, opt=opt, ignore=ignore)[0]
        ntiles = params.get('ntiles', 1)
        psf_s2n = params.get('psf_s2n', None)

        nbands = params['nbands']
        logger.debug("ntiles, nbands, tile_num, band_num = %d, %d, %d, %d",
                     ntiles, nbands, tile_num, band_num)

        # The rest is pretty much the same as the base class (galsim.des.des_meds.MEDSBuilder),
        # except for how the 'id' field is set.
        # In the base id = obj_num + 1, but when doing multiple bands, we need the ids to repeat (obj_num does not).
        # So instead in the loop over nobjects below, use tile_start_obj_num + i.
        if base.get('image', {}).get('type', 'Single') != 'Single':
            raise galsim.GalSimConfigError(
                "MEDS files are not compatible with image type %s." % base['image']['type'])

        nobjects = params['nobjects']
        nstamps_per_object = params['nstamps_per_object']
        ntot = nobjects * nstamps_per_object

        main_images = galsim.config.BuildImages(ntot, base, image_num=image_num,  obj_num=obj_num,
                                                logger=logger)

        # grab list of offsets for cutout_row/cutout_col.
        offsets = galsim.config.GetFinalExtraOutput('meds_get_offset', base, logger)
        # cutout_row/col is the stamp center (**with the center of the first pixel
        # being (0,0)**) + offset
        centers = [0.5*im.array.shape[0]-0.5 for im in main_images]
        cutout_rows = [c+offset.y for c, offset in zip(centers, offsets)]
        cutout_cols = [c+offset.x for c, offset in zip(centers, offsets)]

        weight_images = galsim.config.GetFinalExtraOutput('weight', base, logger)
        if 'badpix' in config:
            badpix_images = galsim.config.GetFinalExtraOutput('badpix', base, logger)
        else:
            badpix_images = None
        psf_images = galsim.config.GetFinalExtraOutput('psf', base, logger)

        # We can optionally add noise to the psf images
        if psf_s2n is not None:
            logger.error("Adding noise to psf images to achieve s/n=%f" % psf_s2n)
            rng = galsim.random.BaseDeviate(seed=obj_num)
            for i, psf in enumerate(psf_images):
                noise = GaussianNoise(rng=rng)
                psf_images[i].addNoiseSNR(
                    noise, psf_s2n, preserve_flux=True)

        obj_list = []
        tile_start_obj_num = base['tile_start_obj_num']
        for i in range(nobjects):
            k1 = i*nstamps_per_object
            k2 = (i+1)*nstamps_per_object
            images_this_obj = main_images[k1:k2]
            weights_this_obj = weight_images[k1:k2]
            psfs_this_obj = psf_images[k1:k2]
            cutout_rows_this_obj = cutout_rows[k1:k2]
            cutout_cols_this_obj = cutout_cols[k1:k2]
            if badpix_images is not None:
                bpk = badpix_images[k1:k2]
            else:
                bpk = None

            obj = MultiExposureObject(images=images_this_obj,
                                      weight=weights_this_obj,
                                      badpix=bpk,
                                      psf=psfs_this_obj,
                                      id=tile_start_obj_num + i,
                                      cutout_row=cutout_rows_this_obj,
                                      cutout_col=cutout_cols_this_obj)
            obj_list.append(obj)

        return obj_list

    def getNFiles(self, config, base):
        """Returns the number of files to be built.
        As far as the config processing is concerned, this is the number of times it needs
        to call buildImages, regardless of how many physical files are actually written to
        disk.  So this corresponds to output.nexp for the FocalPlane output type.
        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @returns the number of "files" to build.
        """
        ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
        nbands = len(config["bands"])
        config["nbands"] = nbands
        return ntiles * nbands


galsim.config.output.RegisterOutputType('MultibandMEDS', MultibandMEDSBuilder())
