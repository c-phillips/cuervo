class classproperty:
    def __init__(self, method):
        self.method = method
    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.method(cls)