import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def load_params(cnmt, filename, expect_partial=False):
    log.info('Loading params from {}'.format(filename))
    cnmt.load_params(filename, expect_partial=expect_partial)
    log.info('END Loading params')


def save_params(cnmt, filename):
    log.info('Saving params to {}'.format(filename))
    cnmt.save_params(filename)
    log.info('END Saving params')
