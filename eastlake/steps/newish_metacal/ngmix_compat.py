# flake8: noqa
import ngmix

if ngmix.__version__[0:2] == "v1":
    NGMIX_V2 = False
    from ngmix.fitting import format_pars

else:
    NGMIX_V2 = True
    from ngmix.util import format_pars
