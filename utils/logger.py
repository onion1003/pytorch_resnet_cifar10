# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

from tensorboardX import SummaryWriter

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir, comment='123')

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        self.flush()

    def text_summary(self, parameters):
        """Log a text variable."""
        self.writer.add_text('parameters', parameters, 0)
        self.flush()

    def flush(self):
        self.writer.file_writer.flush()


if __name__ == '__main__':
    logger_path = ''
    logger = Logger(logger_path)

    # info = {}
    # for tag, value in info.items():
    #     logger.scalar_summary(tag, value, epoch)
    # logger.text_summary("{}".format(args))