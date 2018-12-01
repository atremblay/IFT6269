import logging

class Job:

    def __init__(self, save_file_path, loader, net):
        self.save_file_path = save_file_path
        self.loader = loader
        self.net = net
        self.logger = logging.getLogger(str(type(self)))

    def append_save_data(self, content):
        with open(self.save_file_path, 'a') as f:
            content_format = ', '.join(['{}' for _ in content]) + '\n'
            f.write(content_format.format(*content))